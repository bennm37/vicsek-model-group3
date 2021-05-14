import pygame
from flock import *
from pygame.locals import (
    K_p,
    K_b,
    K_m,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
##colours
BLACK = ((0,0,0))
WHITE =((255,255,255))
BLUE =((0,0,255))
RED =((255,0,0))

##initialize the pygame library
pygame.init()

# set up the drawing window
screen = pygame.display.set_mode([500, 500])
screen_col = WHITE

# run until player quits
running = True

##Setting up the Flock - Change the Parameters to change the number of birds, noise levels and
## type, then run either here or in the notebook
N=400
sigma = 0.3
prey =False
moth = not prey
if prey:
    f= Prey(N,sigma,15)
else:
    f= Moth(N,sigma,15)
Clock = pygame.time.Clock()
while running:

    for event in pygame.event.get():
        ##close if exit button pressed
        if event.type == pygame.QUIT:
            running = False
        if event.type ==KEYDOWN:
            ##change background colour to black
            if event.key ==K_b:
                if screen_col == BLACK:
                    screen_col = WHITE
                else:
                    screen_col = BLACK
            ##change moth to prey
            if event.key ==K_p:
                old_pos =f.positions
                old_thetas = f.thetas
                f = Prey(N,0.3,15)
                f.positions = old_pos
                f.thetas = old_thetas
                prey =True
            ##change prey to moth
            if event.key ==K_m:
                old_pos =f.positions
                old_thetas = f.thetas
                f = Moth(N,0.3,15)
                f.positions = old_pos
                f.thetas = old_thetas
                prey =False
            

    ## fill the background with white
    screen.fill(screen_col)

    ## update predator positions
    predator_pos = np.array([pygame.mouse.get_pos()])
    pygame.draw.circle(screen,(255,0,0),predator_pos[0],10)


    ## update birds positions
    if prey:
        f.update_posdirs(0.1,predator_pos*15/500,repulsion_factor=10)
    elif moth:
        f.update_posdirs(1,predator_pos*15/500,attraction_factor=100)
    else:
        f.update_posdirs(0.1)
    p =np.floor(f.positions*500/15).astype("int32")
    for pos in p:
        pygame.draw.circle(screen,(0,0,255),pos,5)

    # flip the display
    Clock.tick(40)
    pygame.display.flip()

pygame.quit()