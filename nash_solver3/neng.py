import argparse
import time
from game import Game
import logging
import sys
import numpy as np


def parse_args():
    """
    Parse arguments of script.

    :return: parsed arguments
    :rtype: dict
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
NenG - Nash Equilibrium Noncooperative games.
Tool for computing Nash equilibria in noncooperative games.
Specifically:
All pure Nash equilibria in all games (--method=pne).
All mixed Nash equilibria in two-players games (--method=support_enumeration).
One sample mixed Nash equilibria in n-players games (--method={CMAES,L-BFGS-B,SLSQP}).
""",
    )
    pa = parser.add_argument
    pa("-f", "--file", required=True, help="File where game in nfg format is saved.")
    pa(
        "-m",
        "--method",
        default="CMAES",
        choices=Game.METHODS,
        help="Method to use for computing Nash equlibria.",
    )
    pa(
        "-e",
        "--elimination",
        action="store_true",
        default=False,
        help="Use Iterative Elimination of Strictly Dominated Strategies before computing NE.",
    )
    pa(
        "-p",
        "--payoff",
        action="store_true",
        default=True,
        help="Print also players payoff with each Nash equilibrium.",
    )
    pa(
        "-c",
        "--checkNE",
        action="store_true",
        default=False,
        help="After computation check if found strategy profile is really Nash equilibrium.",
    )
    pa(
        "-t",
        "--trim",
        choices=("normalization", "penalization"),
        default="normalization",
        help="Method for keeping strategy profile in probability distribution universum.",
    )
    pa(
        "-l",
        "--log",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Level of logs to save/print",
    )
    pa(
        "--log-file",
        default=None,
        help="Log file. If omitted log is printed to stdout.",
    )
    return parser.parse_args()


def main():
    """
    Main method of script. Based on information from arguments executes needed
    commands.
    """
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log.upper(), None),
        format="%(levelname)s, %(asctime)s, %(message)s",
        filename=args.log_file,
    )

    if args.file == "dic":
        game_str = {
            "name": "Prisoner's dilema",
            "players": ["Peter", "John"],
            "num_players": 2,
            "shape": [2, 2],
            "sum_shape": 4,
            "array": [
                np.asarray([[-10.0, 0.0], [-20.0, -1.0]]),
                np.asarray([[-10.0, -20.0], [0.0, -1.0]]),
            ],
        }
    else:
        with open(args.file) as f:
            game_str = f.read()

    start = time.time()
    g = Game(game_str, args.trim)
    logging.debug("Reading the game took: {0} s".format(time.time() - start))
    if args.elimination:
        g.IESDS()
    result = g.findEquilibria(args.method)
    if result is not None:
        if args.checkNE:
            success = g.checkNEs(result)
        else:
            success = True
        if success:
            print(g.printNE(result, payoff=args.payoff))
        else:
            sys.exit("Nash equilibria did not pass the test.")
    else:
        sys.exit("Nash equilibrium was not found.")


if __name__ == "__main__":
    main()
