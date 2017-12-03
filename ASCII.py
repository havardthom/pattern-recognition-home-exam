import numpy as np

def ASCII(l):
	N = len(l)

	if N % 7 != 0:
		print 'Your label vector is not a multiplum of 7!'
		return

	k = N/7;

	for i in range(k):

		start = i*7;
		stop = start+7;

		bits = l[start:stop]

		if (bits == [-1, -1, -1, -1, -1, 1, 1]).all():
			print(' '),
		elif (bits == [1, -1, -1, -1, -1, 1, 1]).all():
			print('a'),
		elif (bits == [-1, 1, -1, -1, -1, 1, 1]).all():
			print('b'),
		elif (bits == [1, 1, -1, -1, -1, 1, 1]).all():
			print('c'),
		elif (bits == [-1, -1, 1, -1, -1, 1, 1]).all():
			print('d'),
		elif (bits == [1, -1, 1, -1, -1, 1, 1]).all():
			print('e'),
		elif (bits == [-1, 1, 1, -1, -1, 1, 1]).all():
			print('f'),
		elif (bits == [1, 1, 1, -1, -1, 1, 1]).all():
			print('g'),
		elif (bits == [-1, -1, -1, 1, -1, 1, 1]).all():
			print('h'),
		elif (bits == [1, -1, -1, 1, -1, 1, 1]).all():
			print('i'),
		elif (bits == [-1, 1, -1, 1, -1, 1, 1]).all():
			print('j'),
		elif (bits == [1, 1, -1, 1, -1, 1, 1]).all():
			print('k'),
		elif (bits == [-1, -1, 1, 1, -1, 1, 1]).all():
			print('l'),
		elif (bits == [1, -1, 1, 1, -1, 1, 1]).all():
			print('m'),
		elif (bits == [-1, 1, 1, 1, -1, 1, 1]).all():
			print('n'),
		elif (bits == [1, 1, 1, 1, -1, 1, 1]).all():
			print('o'),
		elif (bits == [-1, -1, -1, -1, 1, 1, 1]).all():
			print('p'),
		elif (bits == [1, -1, -1, -1, 1, 1, 1]).all():
			print('q'),
		elif (bits == [-1, 1, -1, -1, 1, 1, 1]).all():
			print('r'),
		elif (bits == [1, 1, -1, -1, 1, 1, 1]).all():
			print('s'),
		elif (bits == [-1, -1, 1, -1, 1, 1, 1]).all():
			print('t'),
		elif (bits == [1, -1, 1, -1, 1, 1, 1]).all():
			print('u'),
		elif (bits == [-1, 1, 1, -1, 1, 1, 1]).all():
			print('v'),
		elif (bits == [1, 1, 1, -1, 1, 1, 1]).all():
			print('w'),
		elif (bits == [-1, -1, -1, 1, 1, 1, 1]).all():
			print('x'),
		elif (bits == [1, -1, -1, 1, 1, 1, 1]).all():
			print('y'),
		elif (bits == [-1, 1, -1, 1, 1, 1, 1]).all():
			print('z'),
		elif (bits == [-1, 1, 1, 1, -1, 1, -1]).all():
			print('.'),
		elif (bits == [1, -1, -1, -1, -1, 1, -1]).all():
			print('!'),
		else:
			print('#'),
	print('\n')
