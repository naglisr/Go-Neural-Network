#!/usr/bin/env python
# coding: utf-8

"""Goban made with Python, pygame and go.py.

This is a front-end for my go library 'go.py', handling drawing and
pygame-related activities. Together they form a fully working goban.

"""

__author__ = "Aku Kotkavuo <aku@hibana.net>"
__version__ = "0.1"

import pygame
import go
from sys import exit
import goAI
import numpy as np

BACKGROUND = 'images/ramin.jpg'
BOARD_SIZE = (820, 820)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class Stone(go.Stone):
    def __init__(self, board, point, color):
        """Create, initialize and draw a stone."""
        super(Stone, self).__init__(board, point, color)
        self.coords = (5 + self.point[0] * 40, 5 + self.point[1] * 40)
        self.draw()

    def draw(self):
        """Draw the stone as a circle."""
        pygame.draw.circle(screen, self.color, self.coords, 20, 0)
        pygame.display.update()

    def remove(self):
        """Remove the stone from board."""
        blit_coords = (self.coords[0] - 20, self.coords[1] - 20)
        area_rect = pygame.Rect(blit_coords, (40, 40))
        screen.blit(background, blit_coords, area_rect)
        pygame.display.update()
        super(Stone, self).remove()


class Board(go.Board):
    def __init__(self):
        """Create, initialize and draw an empty board."""
        super(Board, self).__init__()
        self.outline = pygame.Rect(45, 45, 720, 720)
        self.draw()

    def draw(self):
        """Draw the board to the background and blit it to the screen.

        The board is drawn by first drawing the outline, then the 19x19
        grid and finally by adding hoshi to the board. All these
        operations are done with pygame's draw functions.

        This method should only be called once, when initializing the
        board.

        """
        pygame.draw.rect(background, BLACK, self.outline, 3)
        # Outline is inflated here for future use as a collidebox for the mouse
        self.outline.inflate_ip(20, 20)
        for i in range(18):
            for j in range(18):
                rect = pygame.Rect(45 + (40 * i), 45 + (40 * j), 40, 40)
                pygame.draw.rect(background, BLACK, rect, 1)
        for i in range(3):
            for j in range(3):
                coords = (165 + (240 * i), 165 + (240 * j))
                pygame.draw.circle(background, BLACK, coords, 5, 0)
        screen.blit(background, (0, 0))
        pygame.display.update()

    def update_liberties(self, added_stone=None):
        """Updates the liberties of the entire board, group by group.

        Usually a stone is added each turn. To allow killing by 'suicide',
        all the 'old' groups should be updated before the newly added one.

        """
        for group in self.groups:
            if added_stone:
                if group == added_stone.group:
                    continue
            group.update_liberties()
        if added_stone:
            added_stone.group.update_liberties()


def main():
    while True:
        pygame.time.wait(250)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and board.outline.collidepoint(event.pos):
                    x = int(round(((event.pos[0] - 5) / 40.0), 0))
                    y = int(round(((event.pos[1] - 5) / 40.0), 0))
                    stone = board.search(point=(x, y))
                    if stone:
                        stone.remove()
                    else:
                        added_stone = Stone(board, (x, y), board.turn())
                    board.update_liberties(added_stone)


## NR's stuff all below this line


def play_ai(player_colour, ai_model):
    while True:
        pygame.time.wait(250)
        if board.next == player_colour:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1 and board.outline.collidepoint(event.pos):
                        x = int(round(((event.pos[0] - 5) / 40.0), 0))
                        y = int(round(((event.pos[1] - 5) / 40.0), 0))
                        stone = board.search(point=(x, y))
                        if stone:
                            stone.remove()
                        else:
                            added_stone = Stone(board, (x, y), board.turn())
                        board.update_liberties(added_stone)
        else:
            board_state = np.zeros(362)
            for group in board.groups:
                for s in group.stones:
                    x, y = s.point[0]-1, s.point[1]-1
                    index = 19 * x + y
                    if s.color == BLACK:
                        board_state[index] = 1
                    else:
                        board_state[index] = -1
            if board.next == BLACK:
                board_state[361] = 1

            board_state = board_state.reshape(1, 362)

            prediction = np.array(ai_model(board_state))[0]
            print(prediction.shape)
            found_empty_square = False
            while not found_empty_square:
                movearg = np.argmax(prediction)
                print(movearg)
                move_x = int(movearg/19)
                move_y = movearg - 19*move_x
                stone = board.search(point=(move_x+1, move_y+1))
                if stone:
                    prediction[movearg] = -100
                else:
                    added_stone = Stone(board, (move_x+1, move_y+1), board.turn())
                    found_empty_square = True
            board.update_liberties(added_stone)




def place_stone(x, y):
    stone = board.search(point=(x, y))
    if stone:
        stone.remove()
    else:
        added_stone = Stone(board, (x, y), board.turn())
    board.update_liberties(added_stone)




def wip():
    board_states = goAI.get_x_and_y_data_from_file("/home/naglis/PycharmProjects/goNN/goGames/games/AJ1st/01/1.sgf")

    for state in board_states:
        print(state)

    moves = goAI.process_game_file("/home/naglis/PycharmProjects/goNN/goGames/games/AJ1st/01/1.sgf")
    for move in moves:
        pygame.time.wait(250)
        x = move[0]+1
        y = move[1]+1
        added_stone = Stone(board, (x, y), board.turn())
        board.update_liberties(added_stone)
    #
    # board = Board()

if __name__ == '__main__':
    # aimodel = goAI.train_model("../goGames/games")
    aimodel = goAI.load_model()

    # Coding a move demo
    pygame.init()
    pygame.display.set_caption('Goban')
    screen = pygame.display.set_mode(BOARD_SIZE, 0, 32)
    background = pygame.image.load(BACKGROUND).convert()

    board = Board()
    # wip()
    play_ai(BLACK, aimodel)


    # place_stone(3, 3)
    # stone.draw()
    # main()
