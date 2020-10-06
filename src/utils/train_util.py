import copy
import datetime
import math

import attr
import numpy
import torch

from nsrnn.pytorch_tools.curriculum import RandomShuffling
from nsrnn.pytorch_tools.early_stopping import EarlyStopping
from nsrnn.ticker import TimedTicker

def add_train_arguments(parser):
    group = parser.add_argument_group('Training options')
    group.add_argument('--shuffle-seed', type=int,
        help='Random seed used for random shuffling.')
    add_optimizer_args(group, default='SGD')
    group.add_argument('--learning-rate', type=float, required=True,
        help='Initial learning rate.')
    group.add_argument('--learning-rate-patience', type=int, default=0,
        help='Number of epochs of no improvement before the learning rate is '
             'decreased.')
    group.add_argument('--learning-rate-decay', type=float, default=0.9,
        help='The learning rate will be multiplied by this factor when '
             'patience runs out. Should be between 0 and 1.')
    group.add_argument('--l2-regularizer-lambda', type=float, default=0,
        help='Coefficient for the L2 regularizer.')
    group.add_argument('--gradient-clipping', type=float,
        help='Gradient clipping threshold.')
    group.add_argument('--epochs', type=int, required=True,
        help='Maximum number of epochs to run.')
    group.add_argument('--early-stopping-patience', type=int, default=5,
        help='Number of epochs of no improvement before training stops.')

def parse_optimizer_class(s):
    return getattr(torch.optim, s)

def add_optimizer_args(parser, default='SGD'):
    parser.add_argument('--optimizer', type=parse_optimizer_class,
        default=parse_optimizer_class(default),
        help=f'Type of optimizer to use for training. Default is {default}.')

@attr.s
class TrainState:
    epoch_no = attr.ib()
    update_no = attr.ib()
    batch_no = attr.ib()

def train(args, saver, parameter_groups, train_data, valid_data, vocab,
        model_interface, events, logger):

    model = saver.model
    show_progress = not args.no_progress

    # Seed the RNG used to shuffle the data during training.
    shuffle_generator = numpy.random.RandomState(args.shuffle_seed)

    # Pick some example data to print every once in a while.
    max_printed_batch_size = 5
    printed_batch = valid_data[shuffle_generator.choice(len(valid_data))]
    printed_batch = tuple(x[:max_printed_batch_size] for x in printed_batch)

    # Configure the optimization process.
    OptimizerClass = args.optimizer
    optimizer = OptimizerClass(
        parameter_groups,
        lr=args.learning_rate,
        weight_decay=args.l2_regularizer_lambda
    )
    early_stopping = EarlyStopping(
        'min',
        patience=args.early_stopping_patience
    )
    early_stopping_metric = 'perplexity'
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=early_stopping.mode,
        patience=args.learning_rate_patience,
        factor=args.learning_rate_decay
    )
    learning_curriculum = RandomShuffling(train_data, shuffle_generator)

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

        # Update the learning curriculum (e.g. randomly shuffle the data).
        learning_curriculum.step()

        # Show the current learning rate.
        curr_learning_rate = optimizer.param_groups[0]['lr']
        logger.info(f'  learning rate: {curr_learning_rate}')

        ticker = TimedTicker(len(learning_curriculum.data()), 1)
        train_loss = 0.0
        curr_loss = 0.0
        train_perp_numer = 0.0
        curr_perp_numer = 0.0
        train_num_symbols = 0
        curr_num_symbols = 0

        model.train()
        for batch_no, (x, y) in enumerate(learning_curriculum.data()):
            train_state.batch_no = batch_no
            optimizer.zero_grad()
            # Evaluate the model (forward pass).
            # logits : B x n x V
            logits = model_interface.get_logits(model, x, train_state)
            # Get the loss term for each symbol.
            # y : B x n
            # symbol_losses : B x n
            symbol_losses = criterion(logits.transpose(1, 2), y)
            # Sum over time steps to get the loss for each sequence in the
            # batch.
            # sequence_losses : B
            sequence_losses = torch.sum(symbol_losses, dim=1)
            # Average across batch elements to get the loss.
            loss = torch.mean(sequence_losses, dim=0)
            # Update statistics.
            loss_item = loss.item()
            with torch.no_grad():
                perp_numer = torch.sum(sequence_losses).item()
            num_symbols = y.numel()
            train_loss += loss_item
            curr_loss += loss_item
            train_perp_numer += perp_numer
            curr_perp_numer += perp_numer
            train_num_symbols += num_symbols
            curr_num_symbols += num_symbols
            # Run backprop.
            loss.backward()
            # Gradient clipping.
            if args.gradient_clipping is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.gradient_clipping)
            # Update parameters.
            optimizer.step()
            train_state.update_no += 1
            # Show training progress.
            ticker.progress = batch_no + 1
            if show_progress and ticker.tick():
                avg_curr_loss = curr_loss / curr_num_symbols
                avg_curr_perplexity = math.exp(curr_perp_numer / curr_num_symbols)
                logger.info(
                    f'  {ticker.int_percent}% | '
                    f'loss: {avg_curr_loss:.2f} | '
                    f'perplexity: {avg_curr_perplexity:.2f}')
                curr_loss = 0.0
                curr_perp_numer = 0.0
                curr_num_symbols = 0
        # Summarize training progress for this epoch.
        avg_train_loss = train_loss / train_num_symbols
        avg_train_perplexity = math.exp(train_perp_numer / train_num_symbols)
        logger.info(
            f'  average loss: {avg_train_loss:.2f} | '
            f'average perplexity: {avg_train_perplexity:.2f}')
        events.log('epoch', {
            'loss' : avg_train_loss,
            'perplexity' : avg_train_perplexity
        })

        # Show example outputs from the model.
        model_interface.print_example(model, printed_batch, vocab, logger)

        # Evaluate the model on the validation set.
        valid_scores = evaluate(
            model,
            valid_data,
            model_interface,
            show_progress,
            logger,
            logger_indent='  '
        )
        events.log('validate', valid_scores)
        score = valid_scores[early_stopping_metric]
        logger.info(f'  dev perplexity: {valid_scores["perplexity"]:.3f}')
        logger.info(f'  dev accuracy:   {valid_scores["accuracy"]:.2%}')
        # Adjust the learning rate based on the score.
        if learning_curriculum.done():
            lr_scheduler.step(score)
        is_best, should_stop = early_stopping.update(score)
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

def evaluate(model, batches, model_interface, show_progress, logger,
        logger_indent=''):
    s = logger_indent
    model.eval()
    perp_numer = 0.0
    acc_numer = 0
    num_symbols = 0
    with torch.no_grad():
        ticker = TimedTicker(len(batches), 1)
        for batch_no, (x, y_target) in enumerate(batches):
            # Let B be batch size, n be sequence length, and V be vocabulary size.
            # y_logits : B x n x V
            # y_target : B x n of int in [0, V-1]
            y_logits = model_interface.get_logits(model, x, None)
            # Compute the numerator for the perplexity score.
            # -log(p_M(y_target)) = -\sum_t log(p_M(y_target_{t} | y_target_{<t}))
            y_neg_log_prob = torch.nn.functional.cross_entropy(
                y_logits.transpose(1, 2),
                y_target,
                reduction='sum')
            perp_numer += y_neg_log_prob.item()
            # Compute the numerator for the accuracy score.
            # Get the symbol with the highest score at each timestep.
            y_pred = torch.argmax(y_logits, dim=2)
            num_correct = torch.eq(y_pred, y_target).sum()
            acc_numer += num_correct.item()
            num_symbols += y_target.numel()
            ticker.progress = batch_no + 1
            if show_progress and ticker.tick():
                logger.info(f'{s}{ticker.int_percent}%')
    perplexity = math.exp(perp_numer / num_symbols)
    accuracy = acc_numer / num_symbols
    return {
        'perplexity' : perplexity,
        'accuracy' : accuracy,
        'perplexity_numerator' : perp_numer,
        'accuracy_numerator' : acc_numer,
        'num_symbols' : num_symbols
    }
