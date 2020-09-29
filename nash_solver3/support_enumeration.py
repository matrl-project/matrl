# -*- coding: utf-8 -*-

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

import logging
import itertools

import numpy as np

import nash_solver3.strategy_profile as sp


class SupportEnumeration(object):
    """
    Class providing support enumeration method for finding all mixed Nash
    equilibria in two-players games.
    """

    def __init__(self, game):
        self.game = game

    def getEquationSet(self, combination, player, num_supports):
        r"""
        Return set of equations for given player and combination of strategies
        for 2 players games in support_enumeration

        This function returns matrix to compute (Nisan algorithm 3.4)


        For given :math:`I` (subset of strategies) of player 1 we can write down next
        equations:

        .. math::

            \sum_{i \in I} x_i b_{ij} = v \\
            \sum_{i \in I} x_i = 1

        Where :math:`x_i` is probability of ith strategy, :math:`b_{ij}` is payoff for player
        2 with strategies :math:`i \in I, j \in J`, :math:`v` payoff for player 1

        In matrix form (k = num_supports):

        .. math::

            \begin{pmatrix}
            b_{11} & b_{12} & \cdots & b_{1k} & -1 \\
            b_{21} & b_{22} & \cdots & b_{2k} & -1 \\
            \vdots  & \vdots  & \ddots & \vdots & -1 \\
            b_{k1} & b_{k2} & \cdots & b_{kk} & -1 \\
            1      &    1   & \cdots &  1     &  0
            \end{pmatrix}
            \begin{pmatrix}
            x_1 \\
            x_2 \\
            \vdots \\
            x_k \\
            v
            \end{pmatrix}
            =
            \begin{pmatrix}
            0 \\
            0 \\
            \vdots \\
            0 \\
            1
            \end{pmatrix}

        Analogically for result y for player 2 with payoff matrix A

        :param combination: combination of strategies to make equation set
        :type combination: tuple
        :param player: order of player for whom the equation will be computed
        :type player: int
        :param num_supports: number of supports for player
        :type num_supports: int
        :return: equation matrix for solving in np.linalg.solve
        :rtype: np.array
        """
        row_index = np.zeros(self.game.shape[0], dtype=bool)
        col_index = np.zeros(self.game.shape[1], dtype=bool)
        row_index[list(combination[0])] = True
        col_index[list(combination[1])] = True
        numbers = self.game.array[(player + 1) % 2][row_index][:, col_index]
        last_row = np.ones((1, num_supports + 1))
        last_row[0][-1] = 0
        last_column = np.ones((num_supports, 1)) * -1
        if player == 0:
            numbers = numbers.T
        numbers = np.hstack((numbers, last_column))
        numbers = np.vstack((numbers, last_row))
        return numbers

    def supportEnumeration(self):
        """
        Computes all mixed NE of 2 player noncooperative games.
        If the game is degenerate game.degenerate flag is ticked.

        :return: list of NE computed by method support enumeration
        :rtype: list
        """
        result = []
        # for every numbers of supports
        for num_supports in range(1, min(self.game.shape) + 1):
            logging.debug(
                "Support enumearation for num_supports: {0}".format(num_supports)
            )
            supports = []
            equal = [0] * num_supports
            equal.append(1)
            # all combinations of support length num_supports
            for player in range(self.game.num_players):
                supports.append(
                    itertools.combinations(range(self.game.shape[player]), num_supports)
                )
                # cartesian product of combinations of both player
            for combination in itertools.product(supports[0], supports[1]):
                mne = []
                is_mne = True
                # for both player compute set of equations
                for player in range(self.game.num_players):
                    equations = self.getEquationSet(combination, player, num_supports)
                    try:
                        equations_result = np.linalg.solve(equations, equal)
                    except np.linalg.LinAlgError:  # unsolvable equations
                        is_mne = False
                        break
                    probabilities = equations_result[:-1]
                    # all probabilities are nonnegative
                    if not np.all(probabilities >= 0):
                        is_mne = False
                        break
                    player_strategy_profile = np.zeros(self.game.shape[player])
                    player_strategy_profile[list(combination[player])] = probabilities
                    mne.append(player_strategy_profile)
                    # best response
                if is_mne:
                    mne_flat = list(mne[0][:]) + list(mne[1][:])
                    prof = sp.StrategyProfile(mne_flat, self.game.shape)
                    for player in range(self.game.num_players):
                        if not self.game.isMixedBestResponse(player, prof):
                            is_mne = False
                            break
                if is_mne:
                    result.append(
                        sp.StrategyProfile(
                            [item for sublist in mne for item in sublist],
                            self.game.shape,
                        )
                    )
        return result


def computeNE(game):
    """
    Function for easy calling SupportEnumeration from other modules.

    :return: result of support enumeration algorithm
    :rtype: list of StrategyProfile
    """
    se = SupportEnumeration(game)
    return se.supportEnumeration()
