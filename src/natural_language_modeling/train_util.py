import copy
import datetime
import math
import time

import torch

from torch_extras.early_stopping import UpdatesWithoutImprovement
from lib.pytorch_tools.curriculum import InOrder
from lib.ticker import TimedTicker
from cfl_language_modeling.train_util import add_optimizer_args, TrainState

def scaling_factor(s):
    x = float(s)
    if not (0 < x <= 1):
        raise ValueError(f'must be between 0 and 1')
    return x

def add_train_arguments(parser):
    group = parser.add_argument_group('Training options')
    add_optimizer_args(group, default='Adam')
    group.add_argument('--learning-rate', type=float, required=True,
        help='Initial learning rate, normalized by number of words predicted.')
    group.add_argument('--learning-rate-schedule-type', choices=[
        'epochs-without-improvement',
        'zaremba-2014'
    ], default='epochs-without-improvement')
    group.add_argument('--learning-rate-patience', type=int,
        default=5,
        help='(epochs-without-improvement) '
             'Number of epochs of no improvement before the learning rate is '
             'decreased.')
    group.add_argument('--learning-rate-scaling-factor', type=scaling_factor,
        default=0.5,
        help='(epochs-without-improvement, zaremba-2014) '
             'The factor by which the learning rate is multiplied when it '
             'should be decreased. Should be between 0 and 1.')
    group.add_argument('--learning-rate-delay', type=int,
        default=5,
        help='(zaremba-2014) '
             'Number of epochs to wait before the learning rate is decreased.')
    group.add_argument('--l2-regularizer-lambda', type=float,
        default=0,
        help='Coefficient for the L2 regularizer. Default is 0.')
    group.add_argument('--gradient-clip-threshold', type=float,
        help='Gradient clipping threshold, normalized by number of words '
             'predicted. Default is no gradient clipping.')
    group.add_argument('--epochs', type=int,
        default=200,
        help='Maximum number of epochs to run.')
    group.add_argument('--early-stopping-patience', type=int,
        default=math.inf,
        help='Number of epochs of no improvement before training stops. '
             'Default is infinite.')

def train(args, saver, parameter_groups, train_data, valid_data, vocab,
        model_interface, events, logger):

    model = saver.model
    show_progress = not args.no_progress

    # Pick some example data to print every once in a while.
    max_printed_batch_size = 5
    printed_batch = valid_data[0]
    printed_batch = tuple(x[:max_printed_batch_size] for x in printed_batch)

    # Configure the optimization process.
    OptimizerClass = args.optimizer
    optimizer = OptimizerClass(
        parameter_groups,
        lr=args.learning_rate,
        weight_decay=args.l2_regularizer_lambda
    )

    valid_metric = 'perplexity'
    valid_mode = 'min'
    early_stopping = UpdatesWithoutImprovement(
        valid_mode,
        patience=args.early_stopping_patience
    )
    if args.learning_rate_schedule_type == 'epochs-without-improvement':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=valid_mode,
            patience=args.learning_rate_patience,
            factor=args.learning_rate_scaling_factor
        )
    elif args.learning_rate_schedule_type == 'zaremba-2014':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda i: args.learning_rate_scaling_factor ** max(i - args.learning_rate_delay + 1, 0)
        )
    else:
        raise ValueError
    learning_curriculum = InOrder(train_data)

    # Configure the loss function.
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    train_state = TrainState(epoch_no=None, update_no=0, batch_no=None)

    total_start_time = datetime.datetime.now()
    epoch_no = None
    best_epoch = None
    best_parameters = None
    best_metrics = None
    for epoch_no in range(args.epochs):
        train_state.epoch_no = epoch_no
        epoch_start_time = datetime.datetime.now()
        logger.info(f'epoch #{epoch_no + 1}')

        # Update the learning curriculum.
        learning_curriculum.step()

        # Show the current learning rate.
        curr_learning_rate = optimizer.param_groups[0]['lr']
        logger.info(f'  learning rate: {curr_learning_rate}')

        ticker = TimedTicker(len(learning_curriculum.data()), 1)
        train_loss = 0.0
        train_perp_numer = 0.0
        train_num_symbols = 0
        if show_progress:
            curr_loss = 0.0
            curr_perp_numer = 0.0
            curr_num_symbols = 0

        model.train()
        batch_size = learning_curriculum.data()[0][0].size(0)
        model_state = model_interface.get_initial_state(model, batch_size)
        if show_progress:
            speed_start_time = time.time()
        for batch_no, (x, y) in enumerate(learning_curriculum.data()):
            train_state.batch_no = batch_no
            optimizer.zero_grad()
            # Resize the number of batch elements in the model state if
            # necessary.
            actual_batch_size = x.size(0)
            if model_state is not None and model_state.batch_size() > actual_batch_size:
                model_state = model_state.slice_batch(slice(actual_batch_size))
            # Evaluate the model (forward pass).
            # logits : B x n x V
            logits, model_state = model_interface.get_logits_and_state(saver, model_state, x, train_state)
            model_state = model_state.detach()
            # Get the loss term for each symbol.
            # y : B x n
            # symbol_losses : B x n
            symbol_losses = criterion(logits.transpose(1, 2), y)
            # Average across all symbols to get the loss. This makes the
            # learning rate scale with the sequence length and batch size.
            loss = torch.mean(symbol_losses)
            # Update statistics.
            detached_loss = loss.detach()
            with torch.no_grad():
                perp_numer = torch.sum(symbol_losses.detach())
            num_symbols = y.numel()
            train_loss += detached_loss
            train_perp_numer += perp_numer
            train_num_symbols += num_symbols
            if show_progress:
                curr_loss += detached_loss
                curr_perp_numer += perp_numer
                curr_num_symbols += num_symbols
            # Run backprop.
            loss.backward()
            # Gradient clipping.
            # The gradient clipping threshold is not scaled by the number of
            # predictions here because the amount of gradient is already
            # normalized in the loss function using mean.
            if args.gradient_clip_threshold is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.gradient_clip_threshold)
            # Update parameters.
            optimizer.step()
            train_state.update_no += 1
            # Show training progress.
            ticker.progress = batch_no + 1
            if show_progress and ticker.tick():
                avg_curr_loss = curr_loss / curr_num_symbols
                avg_curr_perplexity = torch.exp(curr_perp_numer / curr_num_symbols)
                words_per_second = curr_num_symbols / (time.time() - speed_start_time)
                avg_curr_loss_item = avg_curr_loss.item()
                avg_curr_perplexity_item = avg_curr_perplexity.item()
                logger.info(
                    f'  {ticker.int_percent}% | '
                    f'loss: {avg_curr_loss_item:.4f} | '
                    f'perplexity: {avg_curr_perplexity_item:.2f} | '
                    f'wps: {words_per_second}')
                curr_loss = 0.0
                curr_perp_numer = 0.0
                curr_num_symbols = 0
                speed_start_time = time.time()
        # Summarize training progress for this epoch.
        avg_train_loss = train_loss / train_num_symbols
        avg_train_perplexity = torch.exp(train_perp_numer / train_num_symbols)
        avg_train_loss_item = avg_train_loss.item()
        avg_train_perplexity_item = avg_train_perplexity.item()
        logger.info(
            f'  average loss: {avg_train_loss_item:.4f} | '
            f'average perplexity: {avg_train_perplexity_item:.2f}')
        events.log('epoch', {
            'loss' : avg_train_loss_item,
            'perplexity' : avg_train_perplexity_item
        })

        # Show example outputs from the model.
        model_interface.print_example(saver, printed_batch, vocab, logger)

        # Evaluate the model on the validation set.
        valid_scores = evaluate(
            saver,
            valid_data,
            model_interface,
            show_progress,
            logger,
            logger_indent='  '
        )
        events.log('validate', valid_scores)
        valid_score = valid_scores[valid_metric]
        logger.info(f'  dev perplexity: {valid_scores["perplexity"]:.3f}')
        logger.info(f'  dev accuracy:   {valid_scores["accuracy"]:.2%}')
        # Adjust the learning rate based on the score.
        if learning_curriculum.done():
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(valid_score)
            else:
                lr_scheduler.step()
        is_best, should_stop = early_stopping.update(valid_score)
        # Save the best model seen so far.
        if is_best:
            if args.save_model:
                logger.info('  saving parameters')
                saver.save()
            best_parameters = copy.deepcopy(model.state_dict())
            best_epoch = epoch_no
            best_metrics = valid_scores
        # Early stopping.
        if should_stop:
            last_epoch = epoch_no
            break
        epoch_duration = datetime.datetime.now() - epoch_start_time
        logger.info(f'  epoch duration: {epoch_duration}')
    else:
        last_epoch = epoch_no
    total_duration = datetime.datetime.now() - total_start_time
    logger.info(f'total duration: {total_duration}')
    # Restore the parameters of the best model.
    if best_parameters is not None:
        model.load_state_dict(best_parameters)
    logger.info(f'best epoch: {best_epoch + 1}')
    if best_metrics is not None:
        logger.info(f'best dev perplexity: {best_metrics["perplexity"]:.3f}')
        logger.info(f'best dev accuracy:   {best_metrics["accuracy"]:.2%}')
    result = {
        'best_epoch' : best_epoch + 1,
        'last_epoch' : last_epoch + 1,
        'best_validation_metrics' : best_metrics,
        'epochs_since_improvement' : early_stopping.updates_since_improvement
    }
    events.log('train', result)
    return result

def evaluate(saver, batches, model_interface, show_progress, logger,
        logger_indent=''):
    s = logger_indent
    model = saver.model
    perp_numer = 0.0
    acc_numer = 0
    num_symbols = 0
    model.eval()
    with torch.no_grad():
        ticker = TimedTicker(len(batches), 1)
        model_state = None
        for batch_no, (x, y_target) in enumerate(batches):
            # Let B be batch size, n be sequence length, and V be vocabulary size.
            # Resize the number of batch elements in the model state if
            # necessary.
            actual_batch_size = x.size(0)
            if model_state is not None and model_state.batch_size() > actual_batch_size:
                model_state = model_state.slice_batch(slice(actual_batch_size))
            # y_logits : B x n x V
            # y_target : B x n of int in [0, V-1]
            y_logits, model_state = model_interface.get_logits_and_state(saver, model_state, x, None)
            # Compute the numerator for the perplexity score.
            # -log(p_M(y_target)) = -\sum_t log(p_M(y_target_{t} | y_target_{<t}))
            y_neg_log_prob = torch.nn.functional.cross_entropy(
                y_logits.transpose(1, 2),
                y_target,
                reduction='sum')
            perp_numer += y_neg_log_prob
            # Compute the numerator for the accuracy score.
            # Get the symbol with the highest score at each timestep.
            y_pred = torch.argmax(y_logits, dim=2)
            num_correct = torch.eq(y_pred, y_target).sum()
            acc_numer += num_correct
            num_symbols += y_target.numel()
            ticker.progress = batch_no + 1
            if show_progress and ticker.tick():
                logger.info(f'{s}{ticker.int_percent}%')
    perplexity = torch.exp(perp_numer / num_symbols)
    accuracy = acc_numer / num_symbols
    return {
        'perplexity' : perplexity.item(),
        'perplexity_numerator' : perp_numer.item(),
        'accuracy' : accuracy.item(),
        'accuracy_numerator' : acc_numer.item(),
        'num_symbols' : num_symbols
    }
