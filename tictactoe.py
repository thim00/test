class TTTGameBoard:
    """ Tic Tac Toe game board class """
    def __init__(self):
        self.state = [""] * 9

    def displayGameBoard(self):
        game_layout = """
         {0:1} | {1:1} | {2:1} 
        -----------
         {3:1} | {4:1} | {5:1} 
        -----------
         {6:1} | {7:1} | {8:1} 
        """
        print(game_layout.format(*self.state))
    
    def populateBoard(self, position, value):
        self.state[position] = value

    def isValidMove(self, move):
        """ Check to see if the given move is valid. """
        if move not in range(9):
            return False
        else:
            return self.state[move] == ""

    def isGameOver(self):
        """ Check to see if the game is over. """
        # Go through each position and see if we have a winner.
        for i, value in enumerate(self.state):
            if value == "":
                continue
            if i == 0:
                if value == self.state[1] and value == self.state[2]:
                    return True, value
                elif value == self.state[3] and value == self.state[6]:
                    return True, value
                elif value == self.state[4] and value == self.state[8]:
                    return True, value
            elif i == 1:
                if value == self.state[4] and value == self.state[7]:
                    return True, value
            elif i == 2:
                if value == self.state[4] and value == self.state[6]:
                    return True, value
                elif value == self.state[5] and value == self.state[8]:
                    return True, value
            elif i == 3:
                if value == self.state[4] and value == self.state[5]:
                    return True, value
            elif i == 6:
                if value == self.state[7] and value == self.state[8]:
                    return True, value

        # Check to see if every game value is filled in.
        for value in self.state:
            if value == "":
                return False, None
        
        # If we get here, we have a tie game.
        return True, None



class TTTGame:
    """ Tic Tac Toe game class """
    def __init__(self, player1, player2):
        self.board = TTTGameBoard()
        self.player1 = player1
        self.player2 = player2

    def play(self):
        """ Play the game until it's over. """
        while True:
            self.board.displayGameBoard()
            position = self.player1.makeMove(self)
            value = self.player1.value
            self.board.populateBoard(position, value)
            game_over, result = self.board.isGameOver()
            if game_over:
                self.board.displayGameBoard()
                if result == None:
                    print("Game over.  Tie game.")
                else:
                    print("Game is over and winner is {0}".format(self.player1.name))
                break

            self.board.displayGameBoard()
            position = self.player2.makeMove(self)
            value = self.player2.value
            self.board.populateBoard(position, value)
            game_over, result = self.board.isGameOver()
            if game_over:
                self.board.displayGameBoard()
                if result == None:
                    print("Game over.  Tie game.")
                else:
                    print("Game is over and winner is {0}".format(self.player2.name))
                break

    def isValidMove(self, move):
        """ Check to see if the given move is valid. """
        return self.board.isValidMove(move)


class TTTPlayer:
    """ Tic Tac Toe player class """
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def makeMove(self, game):
        """ Make a move by entering in a valid open position. """
        while True:
            user_input = input("Enter {0}'s move (value from 0 to 8): ".format(self.name))
            try:
                move = int(user_input)
            except ValueError:
                print("Invalid move.  Please enter a number between 0 and 8.")
                continue
            else:
                if game.isValidMove(move):
                    return move
                else:
                    print("{0} is not a valid move. Please try again.".format(move))





john = TTTPlayer("John", "X")
mary = TTTPlayer("Mary", "O")

new_game = TTTGame(john, mary)
new_game.play()

