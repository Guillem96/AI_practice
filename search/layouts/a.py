import sys
import random



def mgenerator(nr,nc,prob,seed):
	random.seed(seed)
	pacman_pos = (random.randint(1,nr-2), random.randint(1,nc-2))
	food_pos = (random.randint(1,nr-2), random.randint(1,nc-2))
	while pacman_pos == food_pos:
		food_pos = (random.randint(1,nr-2), random.randint(1,nc-2))
	for r in range(nr):
		for c in range(nc):
			if r>0 and r<nr-1 and c>0 and c<nc-1:
				if (r,c) == food_pos:
					sys.stdout.write('.')
				elif (r,c) == pacman_pos:
					sys.stdout.write('P')
				elif random.random() < prob: 
					sys.stdout.write('%')
				else: 
					sys.stdout.write(' ')
			else: 
				sys.stdout.write('%')
		sys.stdout.write('\n')
	
if __name__ == "__main__":
	nr = int (sys.argv[1])
	nc = int (sys.argv[2])
	prob = float (sys.argv[3])
	seed = int (sys.argv[4])
	mgenerator(nr,nc,prob,seed)
