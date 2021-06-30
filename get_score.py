f = open('./logs/base/epoch-metric.txt', 'r')

max_bleu = 0.0
max_bleu1 = 0.0
max_bleu2 = 0.0
max_bleu3 = 0.0
max_bleu4 = 0.0

for line in f:
	if line.startswith('BLEU'):
		bleu_scores = line.strip().split(',')
		bleu = float(bleu_scores[0].split('=')[1].strip())
		if bleu > max_bleu:
			max_bleu = bleu
		bleu1 = float(bleu_scores[1].split('=')[0].strip())
		if bleu1 > max_bleu1:
			max_bleu1 = bleu1
		bleu2 = float(bleu_scores[2].split('=')[0].strip())
		if bleu2 > max_bleu2:
			max_bleu2 = bleu2
		bleu3 = float(bleu_scores[3].split('=')[0].strip())
		if bleu3 > max_bleu3:
			max_bleu3 = bleu3
		bleu4 = float(bleu_scores[4].split('(')[0].strip())
		if bleu4 > max_bleu4:
			max_bleu4 = bleu4

print(max_bleu)
print(max_bleu1)
print(max_bleu2)
print(max_bleu3)
print(max_bleu4)

