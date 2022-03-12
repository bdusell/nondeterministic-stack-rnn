import csv
import sys

def main():
    suite = None
    correct = 0
    total = 0
    for row in csv.DictReader(sys.stdin, delimiter='\t'):
        suite = row['suite']
        result_str = row['result']
        assert result_str in ('True', 'False')
        is_correct = result_str == 'True'
        correct += int(is_correct)
        total += 1
    accuracy = correct / total
    writer = csv.writer(sys.stdout, delimiter='\t')
    writer.writerow([suite, accuracy])

if __name__ == '__main__':
    main()
