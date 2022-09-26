import copy

from pyrsistent import mutant

from player import Player
import random
import numpy as np
import math
import redis
from operator import attrgetter


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        self.redis_ = redis.Redis()
        self.redis_.flushdb()
        self.generation = 1

    def add_noise(self, array):
        random_number = np.random.uniform(0, 1, 1)
        if random_number < 0.2:
            mn = np.min(array)
            mx = np.max(array)
            for i in range(len(array)):
                array[i] += np.random.normal(0, (mx - mn) / 2, array[i].shape)
        return


    def mutate(self, child):

        for layer in range(0,len(child.nn.weights)):
            self.add_noise(child.nn.weights[layer])
        for bias in range(0,len(child.nn.biases)):
            self.add_noise(child.nn.biases[bias])

        return

    def roulette_wheel_selection(self,players,num_players):
        population_fitness = sum([player.fitness for player in players])
        player_probabilities = [player.fitness/population_fitness for player in players]
        return list(np.random.choice(players,num_players,p=player_probabilities))

    def q_selection(self,players,num_players,q):
        new_player = []
        for i in range(0,num_players):
            temp = list(np.random.choice(players,q))
            # print(temp)
            # print(temp[0])
            max_attr = max(temp, key=attrgetter('fitness'))
            new_player.append(max_attr)
        return new_player


    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        rw_q_sus_sort = 2
        if rw_q_sus_sort == 1:
            players = self.roulette_wheel_selection(players,num_players)
        elif rw_q_sus_sort == 2:
            players = self.q_selection(players,num_players,3)
        else:
            players = sorted(players, key=lambda x: x.fitness)
            players.reverse()
        # sorted_players = sorted_players
        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)
        # TODO (Additional: Learning curve)
        avg = 0.0
        min_ = math.inf
        max_ = 0
        for p in players:
            avg += p.fitness
            if max_ < p.fitness:
                max_ = p.fitness
            if min_ > p.fitness:
                min_ = p.fitness
        avg = float(avg)/len(players)

        self.redis_.set(f'{self.generation}:avg',avg)
        self.redis_.set(f'{self.generation}:max',max_)
        self.redis_.set(f'{self.generation}:min',min_)

        self.generation += 1
        return players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        new_players = []
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            # 1)
            # 2)
            # prev_players = sorted(prev_players, key=lambda x: x.fitness)
            # prev_players.reverse()
            
            rw_q_sus_sort = 2
            if rw_q_sus_sort == 1:
                prev_players = self.roulette_wheel_selection(prev_players,len(prev_players))
            elif rw_q_sus_sort == 2:
                prev_players = self.q_selection(prev_players,len(prev_players),2)
            else:
                random.shuffle(prev_players)



            counter = 0
            while counter != len(prev_players):
                parent_1 = prev_players[counter]
                counter += 1
                parent_2 = prev_players[counter]
                counter += 1
                child_1 = self.clone_player(parent_1)
                child_2 = self.clone_player(parent_2)
                for layer in range(0,len(parent_1.nn.weights)):
                    rnd = random.randint(0,len(parent_1.nn.weights[layer])-1)
                    child_1.nn.weights[layer][:rnd] = parent_2.nn.weights[layer][:rnd]
                    child_2.nn.weights[layer][:rnd] = parent_1.nn.weights[layer][:rnd]
                    
                for bias in range(0,len(parent_1.nn.biases)):
                    rnd = random.randint(0,len(parent_1.nn.biases[bias])-1)
                    child_1.nn.biases[bias][:rnd] = parent_2.nn.biases[bias][:rnd]
                    child_2.nn.biases[bias][:rnd] = parent_1.nn.biases[bias][:rnd]
                #mutation
                self.mutate(child_1)
                self.mutate(child_2)
                new_players.append(child_1)
                new_players.append(child_2)

            # new_players = prev_players  # DELETE THIS AFTER YOUR IMPLEMENTATION
            return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player
