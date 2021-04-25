import random
class Card():
    def __init__(self,suit,rank):
        self.suit =suit
        self.rank = rank
SUITS = ["H","D","S","C"]
RANKS = ["A","2","3","4","5","6","7","8","9","10","J","Q","K"]
class Hand():
    def __init__(self,num_cards):
        self.cards = [Card(random.choice(SUITS),random.choice(RANKS)) for i in range (num_cards)]
        self.ranks = [c.rank for c in self.cards]
    def display(self):
        for card in self.cards:
            print(card.rank+card.suit)
    def change_ranks(self):
        self.ranks = ["1" for c in self.cards]
h =Hand(2)
h.display()
h.change_ranks
h.display()



