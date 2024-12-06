stateVector = [[0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1]; [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1]];
[cellxrecord, cellyrecord] = noisy_walker_v4('input.txt', 20, 2.0, 1, 0.5, 0.2, 0.1, 20, 0);
cellxrecord
cellyrecord