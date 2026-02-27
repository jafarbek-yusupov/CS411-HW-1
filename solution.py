import heapq
import copy
import time
import random
from enum import Enum

class HeuristicType(Enum):
    HAMMING   = 'hamming'
    MANHATTAN = 'manhattan'

LOG_INDENTATION = "  "

def log(*args, indent: int = 1, after: bool = False, **kwargs):
    print(LOG_INDENTATION * indent + " ".join(str(a) for a in args), **kwargs)
    if after:
        print()

def logSection(title: str, after: bool = False):
    log("=" * 50, indent=0)
    log(title, indent=1)
    log("=" * 50, indent=0, after=after)

class Node:
    def __init__(
        self, 
        board, 
        parent=None, 
        actionLog=None, 
        heuristic: HeuristicType = HeuristicType.MANHATTAN
    ):
        self.board: Board = board
        self.parent: Node = parent
        self.actionLog: str = actionLog
        self.g = 0 if parent == None else parent.g + 1
        self.h = board.manhattan if heuristic == HeuristicType.MANHATTAN else board.hamming
        self.f = self.g + self.h

    @property
    def path(self) -> list:
        '''
        get path to solution:
        - add current node to list
        - do the same process for parent node
        - repeat until node is null
        - then in the end return reversed list
        '''
        node, pathList = self, []
        while node:
            pathList.append(node)
            node = node.parent
        return list(reversed(pathList))
    
    @property
    def isSolved(self) -> bool:
        return self.board.isSolved

    def __lt__(self, other):
        '''
        less-than (<) operator to compare nodes
        (for choosing cheapest path in a* search / storing in min-heap)
        '''
        return self.f < other.f

class Board:
    def __init__(self, state: list[list[int]], size: int = 3):
        self.state = state
        self.size = size
        self.blankCellPosition: tuple[int, int] = self._findBlankCellPosition()

    @property
    def manhattan(self) -> int:
        '''
        manhattan distance: 
        - distance in terms of steps (horizontal + vertical) 
        node needs to pass to reach goal position
        - resulting h score is sum of all those distances

        in comparison to hamming method, manhattan gives us more information
        which is good for heuristic a* search and in comparison to hamming
        we need to visit less nodes to find best path
        '''
        distance = 0
        for row in range(self.size):
            for column in range(self.size):
                cell = self.state[row][column]
                if cell != 0:
                    goalRow, goalColumn = divmod(cell-1, self.size) # -> (x//y, x%y) 
                    # cell value(number) gets mapped to 0-based idx (thus cell-1) 
                    # and using divmod we get values of goal row, column
                    # exmaple: cell=8, size=3; index=(8-1)=7 -> (7//3, 7%1) -> (row: 2, column: 1) <-- goal pos for cell 8
                    distance += abs(row - goalRow) + abs(column - goalColumn)
        return distance
    
    @property
    def hamming(self) -> int:
        '''
        hamming distance - number of positions where the values differ.
        so the algorithm for case of board cells/tiles (where board - 2D list of integers) is:
        - for position [row, column] compare current state cell with goal state cell at given position
        - - board[row][column] != goal[row][column]
        - if mismatch -> increase counter
        - the resulting counter will be the h score for our search algorithm
        '''
        count = 0
        for row in range(self.size):
            for column in range(self.size):
                cell = self.state[row][column]
                if cell != 0:
                    goalRow, goalColumn = divmod(cell - 1, self.size)
                    if row != goalRow or column != goalColumn:
                        count += 1
        return count
    
    @property
    def neighbors(self) -> list:
        """
        return all possible movements (board states) reachable in one move
        """
        blankCellRow, blankCellColumn = self.blankCellPosition
        results = []
        directions = {
            'UP':    (blankCellRow - 1, blankCellColumn),
            'DOWN':  (blankCellRow + 1, blankCellColumn),
            'LEFT':  (blankCellRow, blankCellColumn - 1),
            'RIGHT': (blankCellRow, blankCellColumn + 1),
        }
        for direction, (newRow, newColumn) in directions.items():
            if 0 <= newRow < self.size and 0 <= newColumn < self.size:
                newState = copy.deepcopy(self.state)
                # swap blank cell with adj cell
                newState[blankCellRow][blankCellColumn],newState[newRow][newColumn] = newState[newRow][newColumn], newState[blankCellRow][blankCellColumn]
                movedCell = self.state[newRow][newColumn]
                logAction = f"move tile {movedCell} {direction}"
                results.append((Board(newState, self.size), logAction))
        return results

    @property
    def isSolved(self) -> bool:
        '''
        if stringified version of the board
        looks like numbers from 1 to size^2 in asc order with 0 (free cell) in the end
        then we reached the goal state
        '''
        N = self.size * self.size
        return str(self) == ''.join(map(str, range(1,N))) + '0'

    @property
    def isSolvable(self) -> bool:
        '''
        solvability check by doing inversion count
        - for odd number dimensions (size % 2 == 1):
        - - even inversions -> solvable
        - - odd -> unsolvable

        - for even number dimensions (size % 2 == 0):
        - - apart from main part of algorith (used in odd size case)
        the blank row position (counted from bottom) also matters 
        (so sum of inversions + blankRowFromBottom should be odd for it to be solvable):
        - - inversions EVEN and blank on ODD row from bottom  -> solvable
        - - inversions ODD and blank on EVEN row from bottom -> solvable
        - - else (both are even) -> unsolvable
        '''
        cellList = [tile for row in self.state for tile in row if tile != 0]
        inversions = sum(
            1
            for i in range(len(cellList))
            for j in range(i + 1, len(cellList))
            if cellList[i] > cellList[j]
        )
        if self.size % 2 == 1:
            # only inversion parity matters
            return inversions % 2 == 0
        else: # even dimensions: blank row position considered as well
            blankRowFromBottom = self.size - self.blankCellPosition[0]
            return (inversions + blankRowFromBottom) % 2 == 1
    
    def _findBlankCellPosition(self) -> (tuple[int, int] | None):
        for row in range(self.size):
            for column in range(self.size):
                if self.state[row][column] == 0:
                    return (row, column)
                
    def print(self, label: str = "", indent: int = 1):
        pad    = LOG_INDENTATION * indent
        cellWidth = max(len(str(self.size * self.size)), 2)
        
        # row separators
        top    = "┌" + ("─" * (cellWidth + 2) + "┬") * (self.size - 1) + "─" * (cellWidth + 2) + "┐"
        mid    = "├" + ("─" * (cellWidth + 2) + "┼") * (self.size - 1) + "─" * (cellWidth + 2) + "┤"
        bottom = "└" + ("─" * (cellWidth + 2) + "┴") * (self.size - 1) + "─" * (cellWidth + 2) + "┘"

        if label: print(pad + f"[{label}]")
        print(pad + top)
        for i, row in enumerate(self.state):
            cells = "│".join(
                f" {str(cell).center(cellWidth)} " if cell != 0
                else f" {'*'.center(cellWidth)} " # blank shown as *
                for cell in row
            )
            print(pad + "│" + cells + "│")
            print(pad + (mid if i < self.size - 1 else bottom))

    def __str__(self):
        return ''.join(str(cell) for row in self.state for cell in row)

class SolutionResult:
    def __init__(self, 
        solvable: bool = False,  
        visitedNodeCount: int = 0, 
        timeMs: int = 0,
        pathList: list = None,
    ):
        self.solvable = solvable
        self.solutionPath: list = pathList or []
        self.visitedNodeCount = visitedNodeCount
        self.timeMs = timeMs

    def display(self, showSteps: bool = True):
        if not self.solvable:
            log(f"[UNSOLVABLE] - impossible to reach the goal", after=True)
            return

        moves = len(self.solutionPath) - 1
        log(f"[SOLVED] in {moves} moves | {self.visitedNodeCount} nodes visited | {self.timeMs:.3f}ms", after=True)
        
        if not showSteps: return

        for i, node in enumerate(self.solutionPath):
            if i == 0:
                label = "initial state"
            elif i == moves:
                label = f"[GOAL_STATE]: step {i} - {node.actionLog} == goal state reached"
            else:
                label = f"step {i} - {node.actionLog}"
            node.board.print(label)
            log(f"g={node.g}  h={node.h}  f={node.f}", indent=3)

def solve(board: Board, heuristic: HeuristicType = HeuristicType.MANHATTAN) -> SolutionResult:
    """
    main method to solve the 8 puzzle using a* graph search
    """
    def _getPreciseCurrentTime() -> float:
        return time.perf_counter()

    def _convertToMsAndRoundTime(timestamp: float, precisionDigits = 3) -> float:
        return round(timestamp*1000, precisionDigits)
    
    if not board.isSolvable: return SolutionResult()
    startNode = Node(board, heuristic=heuristic)
    nodeMinHeap: list[Node] = []
    heapq.heappush(nodeMinHeap, startNode)

    # graph search: skip alr visited states/nodes to prevent infinite loops
    visited = set()
    visitedCount = 0
    t0 = _getPreciseCurrentTime()

    while nodeMinHeap:
        current: Node = heapq.heappop(nodeMinHeap)
        if str(current.board) in visited:
            continue

        visited.add(str(current.board))
        visitedCount += 1

        if current.isSolved:
            return SolutionResult(
                solvable=True, 
                visitedNodeCount=visitedCount, 
                timeMs=_convertToMsAndRoundTime(_getPreciseCurrentTime() - t0), 
                pathList=current.path
            )

        for possibleState, actionLog in current.board.neighbors:
            if str(possibleState) not in visited:
                child = Node(
                    possibleState, 
                    parent=current,
                    actionLog=actionLog, 
                    heuristic=heuristic
                )
                heapq.heappush(nodeMinHeap, child)
    return SolutionResult(visitedNodeCount=visitedCount, timeMs=_convertToMsAndRoundTime(_getPreciseCurrentTime() - t0))

class Tester:
    @staticmethod
    def randomSolvableBoard(size: int = 3, randMoveCount: int = 200) -> Board:
        """
        generate a random solvable board by starting from the
        goal state and making random valid moves
        this way we can guarantee solvability - we can never shuffle
        into an unsolvable state by making legal moves
        """
        state = [[(row * size) + col + 1 for col in range(size)]
                 for row in range(size)]
        state[size - 1][size - 1] = 0
        board = Board(state, size)
        for i in range(randMoveCount):
            neighbors = board.neighbors
            board, i = random.choice(neighbors)
        return board

    @staticmethod
    def randomUnsolvableBoard(size: int = 3) -> Board:
        """
        generate a guaranteed unsolvable board by:
        - generating a solvable board
        - swapping two non-blank tiles thus flipping inversion parity -> unsolvable
        - - if solvable guaranteed to have even inversion now its odd
        """
        board = Tester.randomSolvableBoard(size)
        state = copy.deepcopy(board.state)

        pairOfTwoNonBlankTiles = [
            (row, column)
            for row in range(size)
            for column in range(size)
            if state[row][column] != 0
        ]
        (r1, c1), (r2, c2) = pairOfTwoNonBlankTiles[0], pairOfTwoNonBlankTiles[1]
        state[r1][c1], state[r2][c2] = state[r2][c2], state[r1][c1]
        return Board(state, size)

    @staticmethod
    def runCase(
        label: str, 
        board: Board,
        heuristic: HeuristicType = HeuristicType.MANHATTAN,
        showSteps: bool = True
    ):
        logSection(f"{label}  [{heuristic.value}]")
        board.print("initial board", indent=1)
        solve(board, heuristic).display(showSteps=showSteps)

    @staticmethod
    def runAll():
        logSection("[TEST] random 3x3 test cases", after=True)
        for i in range(3):
            Tester.runCase(f"random solvable #{i+1}",   Tester.randomSolvableBoard(size=3))
        for i in range(3):
            Tester.runCase(f"random unsolvable #{i+1}", Tester.randomUnsolvableBoard(size=3))

        logSection("bigger board tests (steps hidden for brevity)", after=True)
        for size in [4, 5]:
            for i in range(3):
                label = f"random solvable {size}x{size} #{i+1}"
                board = Tester.randomSolvableBoard(size=size, randMoveCount=30)
                # showSteps=True to see step by step logs
                Tester.runCase(label, board, showSteps=False) 
                Tester.runCase(label, board, heuristic=HeuristicType.HAMMING, showSteps=False)
            for i in range(3):
                label = f"random unsolvable {size}x{size} #{i+1}"
                # showSteps=True to see step by step logs
                board = Tester.randomUnsolvableBoard(size=size)
                # no point in running test for both heuristic types
                # given that both use the same isSolvable checker
                Tester.runCase(label, board, showSteps=False)

        logSection("[RESULTS] heuristics comparison (for same 3x3 puzzle)")
        testBoard = Board([[1, 2, 5], [3, 4, 0], [6, 7, 8]])
        for h in HeuristicType:
            result = solve(testBoard, heuristic=h)
            log(f"{h.value:<12} -> {result.visitedNodeCount:>5} nodes | {result.timeMs:.3f}ms")

if __name__ == '__main__':
    Tester.runAll()
