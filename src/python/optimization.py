import random
from numpy.linalg import norm
from numpy import dot

def subgradient_optimization(W, fvs, person_to_indices, eta=0.01, iterations=1000000):
    b = 0
    for i in xrange(iterations):
        if random.rand() < 0.5
            while True:
                person = random.choice(person_to_indices)
                if len(person_to_indices[person]) == 1:
                    continue
                i, j = random.sample(person_to_indices[person], 2)
                y = -1
                break
        else:
            first_person, second_person = random.sample(person_to_indices, 2)
            i = random.choice(person_to_indices[first_person])
            j = random.choice(person_to_indices[second_person])
            y = +1

        fv_diff = fvs(:, i) - fvs(:, j)
        dist = norm(dot(W, fv_diff), 2) ** 2

        # update W & b
        if y * (b - dist) < 1:
            W -= eta * y * dot(dot(W, fv_diff), fv_diff.T)
            b += (y + dist - b) / (i + 1)     # iterative mean update.

    return W
