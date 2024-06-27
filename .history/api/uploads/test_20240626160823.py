import random

class Ludo:
    def __init__(self):
        self.board = [None] * 52
        self.safe_spots = [0, 8, 13, 21, 26, 34, 39, 47]
        self.players = {1: {'tokens': [-1, -1, -1, -1], 'start': 0},
                        2: {'tokens': [-1, -1, -1, -1], 'start': 13},
                        3: {'tokens': [-1, -1, -1, -1], 'start': 26},
                        4: {'tokens': [-1, -1, -1, -1], 'start': 39}}

    def roll_dice(self):
        return random.randint(1, 6)

    def is_safe_spot(self, position):
        return position in self.safe_spots

    def get_new_position(self, start, steps):
        return (start + steps) % 52

    def can_move_out(self, player_id):
        return self.players[player_id]['tokens'].count(-1) > 0

    def move_token(self, player_id, token_index, steps):
        start = self.players[player_id]['tokens'][token_index]
        if start == -1:  # Move out of start
            self.players[player_id]['tokens'][token_index] = self.players[player_id]['start']
        else:
            new_position = self.get_new_position(start, steps)
            self.players[player_id]['tokens'][token_index] = new_position

    def get_valid_moves(self, player_id, dice_roll):
        valid_moves = []
        for index, token in enumerate(self.players[player_id]['tokens']):
            if token == -1 and dice_roll == 6:  # Can move out of start
                valid_moves.append((index, "move_out"))
            elif token != -1:
                new_position = self.get_new_position(token, dice_roll)
                valid_moves.append((index, new_position))
        return valid_moves

    def choose_move(self, player_id, dice_roll):
        valid_moves = self.get_valid_moves(player_id, dice_roll)
        if not valid_moves:
            return None
        # Prioritize moving out tokens
        for move in valid_moves:
            if move[1] == "move_out":
                return move
        # Prioritize capturing opponent's tokens
        for move in valid_moves:
            if move[1] != "move_out" and self.board[move[1]] and self.board[move[1]] != player_id:
                return move
        # Prioritize safe spots
        for move in valid_moves:
            if self.is_safe_spot(move[1]):
                return move
        # Default to first valid move
        return valid_moves[0]

    def play_turn(self, player_id):
        dice_roll = self.roll_dice()
        move = self.choose_move(player_id, dice_roll)
        if move:
            token_index, new_position = move
            self.move_token(player_id, token_index, dice_roll)
            print(f"Player {player_id} moves token {token_index} to {self.players[player_id]['tokens'][token_index]}")
        else:
            print(f"Player {player_id} cannot move")

# Example of how to play
ludo_game = Ludo()
for i in range(10):
    for player in ludo_game.players.keys():
        ludo_game.play_turn(player)
