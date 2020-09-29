from operator import mul
import logging

import numpy as np
import scipy.optimize

import nash_solver3.cmaes as cmaes
import nash_solver3.support_enumeration as support_enumeration
import nash_solver3.strategy_profile as sp
import nash_solver3.game_reader as game_reader
from functools import reduce


class Game(object):

    """
    Class Game wrap around all informations of noncooperative game. Also
    it provides basic analyzation of game, like pureBestResponse, if the game
    is degenerate. It also contains an algorithm for iterative elimination
    of strictly dominated strategies and can compute pure Nash equilibria
    using brute force.

    usage:
        >>> g = Game(game_str)
        >>> ne = g.findEquilibria(method='pne')
        >>> print g.printNE(ne)
    """

    METHODS = ["L-BFGS-B", "SLSQP", "CMAES", "support_enumeration", "pne"]

    def __init__(self, nfg, trim="normalization"):
        """
        Initialize basic attributes in Game

        :param nfg: string containing the game in nfg format
        :type nfg: str
        :param trim: method of assuring that strategy profile lies in Delta space,'normalization'|'penalization'
        :type trim: str
        """
        self.num_players = 0
        self.array = None
        self.shape = None
        self.name = ""
        self.players = None
        if type(nfg) is dict:
            game_info = nfg
            self.__dict__.update(nfg)
        else:
            game_info = game_reader.read(nfg)
        self.__dict__.update(game_info)
        print("game_info", game_info)
        self.deltaAssuranceMethod = trim
        self.players_zeros = np.zeros(self.num_players)
        self.brs = None
        self.degenerate = None
        self.deleted_strategies = None

    def pureBestResponse(self, player, strategy):
        """
        Computes pure best response strategy profile for given opponent strategy
        and player

        :param player: player who should respond
        :type player: int
        :param strategy: opponnet strategy
        :type strategy: list
        :return: set of best response strategies
        :rtype: set of coordinates
        """
        strategy = list(strategy)
        result = set()
        strategy[player] = slice(None)  # all possible strategies for 'player'
        payoffs = self.array[player][strategy]
        max_payoff = np.max(payoffs)
        # numbers of best responses strategies
        brs = [index for index, br in enumerate(payoffs) if br == max_payoff]
        for br in brs:
            s = strategy[:]
            s[player] = br
            # made whole strategy profile, not just one strategy
            result.add(tuple(s))
        return result

    def isMixedBestResponse(self, player, strategy_profile):
        """
        Check if strategy of player from strategy_profile is best response
        for opponent strategies.

        :param player: player who should respond
        :type player: int
        :param strategy_profile: strategy profile
        :type strategy: StrategyProfile
        :return: True if strategy_profile[players] is best response
        :rtype: bool
        """
        player_support = np.nonzero(strategy_profile[player])[0]
        payoffs = np.empty_like(strategy_profile[player])
        for strategy in range(self.shape[player]):
            payoffs[strategy] = self.payoff(strategy_profile, player, strategy)
        maximum = np.max(payoffs)
        br = np.where(np.abs(payoffs - maximum) < 1e-4)[0]
        if np.all(player_support == br):
            return True
        else:
            return False

    def getPNE(self):
        """
        Function computes pure Nash equlibria using brute force algorithm.

        :return: list of StrategyProfile that are pure Nash equilibria
        :rtype: list
        """
        self.brs = [set() for i in range(self.num_players)]
        for player in range(self.num_players):
            p_view = self.shape[:]
            p_view[player] = 1
            # get all possible opponent strategy profiles to 'player'
            for strategy in np.ndindex(*p_view):
                # add to list of best responses
                self.brs[player].update(self.pureBestResponse(player, strategy))
                # check degeneration of a game
        self.degenerate = self.isDegenerate()
        # PNE is where all player have best response
        ne_coordinates = set.intersection(*self.brs)
        result = [
            sp.StrategyProfile(coordinate, self.shape, coordinate=True)
            for coordinate in ne_coordinates
        ]
        return result

    def getDominatedStrategies(self):
        """
        :return: list of dominated strategies per player
        :rtype: list
        """
        empty = [slice(None)] * self.num_players
        result = []
        for player in range(self.num_players):
            s1 = empty[:]
            strategies = []
            dominated_strategies = []
            for strategy in range(self.shape[player]):
                s1[player] = strategy
                strategies.append(self.array[player][s1])
            for strategy in range(self.shape[player]):
                dominated = False
                for strategy2 in range(self.shape[player]):
                    if strategy == strategy2:
                        continue
                    elif (strategies[strategy] < strategies[strategy2]).all():
                        dominated = True
                        break
                if dominated:
                    dominated_strategies.append(strategy)
            result.append(dominated_strategies)
        return result

    def IESDS(self):
        """
        Iterative elimination of strictly dominated strategies.

        Eliminates all strict dominated strategies, preserve self.array and
        self.shape in self.init_array and self.init_shape. Stores numbers of
        deleted strategies in self.deleted_strategies. Deletes strategies
        from self.array and updates self.shape.
        """
        self.init_array = self.array[:]
        self.init_shape = self.shape[:]
        self.deleted_strategies = [
            np.array([], dtype=int) for player in range(self.num_players)
        ]
        dominated_strategies = self.getDominatedStrategies()
        while sum(map(len, dominated_strategies)) != 0:
            logging.debug(
                "Dominated strategies to delete: {0}".format(dominated_strategies)
            )
            for player, strategies in enumerate(dominated_strategies):
                for p in range(self.num_players):
                    self.array[p] = np.delete(self.array[p], strategies, player)
                for strategy in strategies:
                    original_strategy = strategy
                    while original_strategy in self.deleted_strategies[player]:
                        original_strategy += 1
                    self.deleted_strategies[player] = np.append(
                        self.deleted_strategies[player], original_strategy
                    )
                self.shape[player] -= len(strategies)
            self.sum_shape = sum(self.shape)
            dominated_strategies = self.getDominatedStrategies()
        for player in range(self.num_players):
            self.deleted_strategies[player].sort()

    def isDegenerate(self):
        """
        Degenerate game is defined for two-players games and there can be
        infinite number of mixed Nash equilibria.

        :return: True if game is said as degenerated
        :rtype: bool
        """
        if self.num_players != 2:
            return False
        if self.brs is None:
            self.getPNE()
        num_brs = [len(x) for x in self.brs]
        num_strategies = [
            reduce(mul, self.shape[:k] + self.shape[(k + 1) :])
            for k in range(self.num_players)
        ]
        if num_brs != num_strategies:
            logging.warning("Game is degenerate.")
            return True
        else:
            return False

    def LyapunovFunction(self, strategy_profile_flat):
        r"""
        Lyapunov function. If LyapunovFunction(p) == 0 then p is NE.

        .. math::

            x_{ij}(p)           & = u_{i}(si, p_i) \\
            y_{ij}(p)           & = x_{ij}(p) - u_i(p) \\
            z_{ij}(p)           & = \max[y_{ij}(p), 0] \\
            LyapunovFunction(p) & = \sum_{i \in N} \sum_{1 \leq j \leq \mu} z_{ij}(p)^2

        Beside this function we need that strategy_profile is in universum
        Delta (basicaly to have character of probabilities for each player).
        We can assure this with two methods: normalization and penalization.

        :param strategy_profile_flat: list of parameters to function
        :type strategy_profile_flat: list
        :return: value of Lyapunov function in given strategy profile
        :rtype: float
        """
        v = 0.0
        acc = 0
        strategy_profile = sp.StrategyProfile(strategy_profile_flat, self.shape)
        if self.deltaAssuranceMethod == "normalization":
            strategy_profile.normalize()
        else:
            strategy_profile_repaired = np.clip(strategy_profile_flat, 0, 1)
            out_of_box_penalty = np.sum(
                (strategy_profile_flat - strategy_profile_repaired) ** 2
            )
            v += out_of_box_penalty
        for player in range(self.num_players):
            u = self.payoff(strategy_profile, player)
            if self.deltaAssuranceMethod == "penalization":
                one_sum_penalty = (1 - np.sum(strategy_profile[player])) ** 2
                v += one_sum_penalty
            acc += self.shape[player]
            for pure_strategy in range(self.shape[player]):
                x = self.payoff(strategy_profile, player, pure_strategy)
                z = x - u
                g = max(z, 0.0)
                v += g ** 2
        return v

    def payoff(self, strategy_profile, player, pure_strategy=None):
        """
        Function to compute payoff of given strategy_profile.

        :param strategy_profile: strategy profile of all players
        :type strategy_profile: StrategyProfile
        :param player: player for whom the payoff is computed
        :type player: int
        :param pure_strategy: if not None player strategy will be replaced by pure strategy of that number
        :type pure_strategy: int
        :return: value of payoff
        :rtype: float
        """
        sp = strategy_profile.copy()
        if pure_strategy is not None:
            sp.updateWithPureStrategy(player, pure_strategy)
        # make product of each probability, returns num_players-dimensional
        # array
        product = reduce(lambda x, y: np.tensordot(x, y, 0), sp)
        result = np.sum(product * self.array[player])
        return result

    def findEquilibria(self, method="CMAES"):
        """
        Find all equilibria, using method

        :param method: of computing equilibria
        :type method: str, one of Game.METHODS
        :return: list of NE, if not found returns None
        :rtype: list of StrategyProfile
        """
        if method == "pne":
            result = self.getPNE()
            if len(result) == 0:
                return None
            else:
                return result
        elif self.num_players == 2 and method == "support_enumeration":
            result = support_enumeration.computeNE(self)
            self.degenerate = self.isDegenerate()
            if len(result) == 0:
                return None
            else:
                return result
        elif method == "CMAES":
            result = cmaes.fmin(self.LyapunovFunction, self.sum_shape)
        elif method in self.METHODS:
            result = scipy.optimize.minimize(
                self.LyapunovFunction,
                np.random.rand(self.sum_shape),
                method=method,
                tol=1e-10,
                options={"maxiter": 1e3 * self.sum_shape ** 2},
            )
        logging.info(result)
        if result.success:
            r = [sp.StrategyProfile(result.x, self.shape)]
            return r
        else:
            return None

    def __str__(self):
        """
        Output in nfg payoff format.

        :return: game in nfg payoff format
        :rtype: str
        """
        result = "NFG 1 R "
        result += '"' + self.name + '"\n'
        result += "{ "
        result += " ".join(['"' + x + '"' for x in self.players])
        result += " } { "
        result += " ".join(map(str, self.shape))
        result += " }\n\n"
        it = np.nditer(self.array[0], order="F", flags=["multi_index", "refs_ok"])
        payoffs = []
        while not it.finished:
            for player in range(self.num_players):
                payoffs.append(self.array[player][it.multi_index])
            it.iternext()
        result += " ".join(map(str, payoffs))
        return result

    def printNE(self, nes, payoff=False):
        """
        Print Nash equilibria with with some statistics

        :param nes: list of Nash equilibria
        :type nes: list of StrategyProfile
        :param payoff: flag to print payoff of each player
        :type payoff: bool
        :return: string to print
        :rtype: str
        """
        lines = []
        for ne in nes:
            ne_copy = ne.copy().normalize()
            self._addDeletedStrategies(ne_copy)
            lines.append("NE " + str(ne_copy))
            if payoff:
                s = [
                    (
                        "{0}: {1:.3f}".format(
                            self.players[player], self.payoff(ne_copy, player)
                        )
                    )
                    for player in range(self.num_players)
                ]
                lines.append("Payoff> " + ", ".join(s))
        return "\n".join(lines)

    def _addDeletedStrategies(self, sp):
        """
        Assure that strategy profile is in same shape as in the beginning.
        It could change because of IESDS and deleted strategies. If so add
        strategies with zero probability.

        :param sp: strategy profile to potentially extend
        :type sp: StrategyProfile
        :return: changed strategy profiles
        :rtype: StrategyProfile
        """
        if self.deleted_strategies is not None:
            for player in range(self.num_players):
                for deleted_strategy in self.deleted_strategies[player]:
                    sp[player].insert(deleted_strategy, 0.0)
                    sp._shape[player] += 1
        return sp

    def checkNEs(self, nes):
        """
        Check if given container of strategy profiles contains only Nash
        equlibria.

        :param nes: container of strategy profiles to examine
        :type list: iterable of strategy profiles
        :return: whether every strategy profile pass NE test
        :rtype: boool
        """
        result = True
        for sp in nes:
            sp_copy = sp.copy().normalize()
            if not self.checkBestResponses(sp_copy):
                result = False
                logging.warning(
                    "Strategy profile {0} is not a Nash equilibrium.".format(sp_copy)
                )
        if result:
            logging.info("Nash equilibrium test passed for all strategy profiles.")
        return result

    def checkBestResponses(self, strategy_profile):
        """
        Check if every strategy of strategy profile is best response to other
        strategies.

        :param strategy_profile: examined strategy profile
        :type strategy_profile: StrategyProfile
        :return: whether every strategy is best response to others
        :rtype: bool
        """
        for player in range(self.num_players):
            if not self.isMixedBestResponse(player, strategy_profile):
                return False
        return True
