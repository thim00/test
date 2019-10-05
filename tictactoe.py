class TTTGameBoard:
    """ Tic Tac Toe game board class """
    def __init__(self):
        self.state = ["", "", "", "", "", "", "", "", ""]

    def displayGameBoard(self):
        game_layout = """
         {0:1} | {1:1} | {2:1} 
        -----------
         {3:1} | {4:1} | {5:1} 
        -----------
         {6:1} | {7:1} | {8:1} 
        """
        print(game_layout.format(self.state[0], self.state[1], self.state[2], self.state[3], self.state[4], self.state[5], self.state[6], self.state[7], self.state[8]))
    
    def populateBoard(self, position, value):
        self.state[position] = value



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
            position = self.player1.makeMove()
            value = self.player1.value
            self.board.populateBoard(position, value)
            if self.isGameOver():
                break

            self.board.displayGameBoard()
            position = self.player2.makeMove()
            value = self.player2.value
            self.board.populateBoard(position, value)
            if self.isGameOver():
                break

        self.board.displayGameBoard()

    def isGameOver(self):
        """ Check to see if the game is over. """
        for i in self.board.state:
            if i == "":
                return False

        return True


class TTTPlayer:
    """ Tic Tac Toe player class """
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def makeMove(self):
        return int(input("Enter player {0} position: ".format(self.name)))




john = TTTPlayer("John", "X")
mary = TTTPlayer("Mary", "O")

new_game = TTTGame(john, mary)
new_game.play()

