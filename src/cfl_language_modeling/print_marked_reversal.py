import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=1)
    parser.add_argument('--pattern', choices=['rainbow', 'barcode', 'increasing-barcode'], default='rainbow')
    args = parser.parse_args()

    w = []
    if args.pattern in ('rainbow', 'barcode'):
        while len(w) < args.length:
            for i in range(args.start, args.stop+1):
                repetitions = i - args.start + 1 if args.pattern == 'barcode' else 1
                for j in range(repetitions):
                    w.append(str(i))
    elif args.pattern == 'increasing-barcode':
        repetitions = 1
        while len(w) < args.length:
            for i in range(args.start, args.stop+1):
                for j in range(repetitions):
                    w.append(str(i))
            repetitions += 1
    w = w[:args.length]
    tokens = [*w, '#', *reversed(w), '</s>']
    print(' '.join(tokens))

if __name__ == '__main__':
    main()
