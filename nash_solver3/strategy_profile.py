#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# Copyright (C) 2013 Petr Šebek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHERWISE
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np


class StrategyProfile(object):

    """
    Wraps information about strategy profile of game.
    """

    def __init__(self, profile, shape, coordinate=False):
        """
        :param profile: one- level list of probability coefficients
        :type profile: list
        :param shape: list of number of strategies per player
        :type shape: list
        :param coordinate: if True, then profile is considered as coordinate in game universum (depict pure strategy profile)
        :type coordinate: bool
        """
        self.shape = shape
        self._list = []
        if coordinate:
            self._coordinateToDeep(profile)
        else:
            self._flatToDeep(profile)

    def _coordinateToDeep(self, coordinate):
        """
        Convert coordinate to deep strategy profile

        :param coordinate: list of numbers to convert
        :type coordinate: list
        """
        for player in range(len(self.shape)):
            self._list.append(np.zeros(self.shape[player]))
            self._list[player][coordinate[player]] = 1.0

    def _flatToDeep(self, profile):
        """
        Convert strategy_profile to deep strategy profile.
        It means that instead of list of length sum_shape we have got nested
        list of length num_players and inner arrays are of shape[player] length

        :param profile: strategy profile to convet
        :type profile: list
        :return: self
        """
        offset = 0
        for player, i in enumerate(self.shape):
            strategy = profile[offset : offset + i]
            self._list.append(np.array(strategy))
            offset += i
        return self

    def normalize(self):
        """
        Normalizes values in StrategyProfile, values can't be negative,
        bigger than one and sum of values of one strategy has to be 1.0.

        :return: self
        """
        for player, strategy in enumerate(self._list):
            self._list[player] = np.abs(strategy) / np.sum(np.abs(strategy))
        return self

    def copy(self):
        """
        Copy constructor for StrategyProfile. Copies content of self to new object.

        :return: StrategyProfile with same attributes
        """
        other = object.__new__(StrategyProfile)
        other._list = [x.copy() for x in self._list]
        other.shape = self.shape[:]
        return other

    def randomize(self):
        """
        Makes strategy of every player random.

        :return: self
        """
        for player in range(len(self.shape)):
            self.randomizePlayerStrategy(player)
        return self

    def randomizePlayerStrategy(self, player):
        """
        Makes strategy of player random.

        :param player: player, whos strategy will be randomized
        :type player: int
        :return: self
        """
        self._list[player] = np.random.rand(self.shape[player])
        return self

    def updateWithPureStrategy(self, player, pure_strategy):
        """
        Replaces strategy of player with pure_strategy

        :param player: order of player
        :type player: int
        :param pure_strategy: order of strategy to be pure
        :type pure_strategy: int
        :return: self
        """
        self._list[player] = np.zeros_like(self._list[player])
        self._list[player][pure_strategy] = 1.0
        return self

    def __str__(self):
        result = ""
        flat_profile = [item for sublist in self._list for item in sublist]
        result += ", ".join(map(str, flat_profile))
        return result

    def __repr__(self):
        return self._list.__repr__()

    def __setitem__(self, key, value):
        if self.shape[key] != len(value):
            raise IndexError(
                "Strategy has to be same length as corresponding shape value is."
            )
        else:
            self._list.__setitem__(key, value)

    def __getitem__(self, item):
        return self._list.__getitem__(item)

    def __eq__(self, other):
        if len(self._list) != len(other._list):
            return False
        for self_player, other_player in zip(self._list, other._list):
            if (self_player != other_player).any():
                return False
        return True
