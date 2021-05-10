import pygame
from flock import *
from pygame.locals import (
    K_p,
    K_b,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
##Colours
BLACK = ((0,0,0))
WHITE =((255,255,255))
BLUE =((0,0,255))
RED =((255,0,0))

#Initialize the pygame library
pygame.init()


# Set up the drawing window
screen = pygame.display.set_mode([500, 500])
screen_col = WHITE

# Run until player quits
running = True

##Setting up the Flock
N=500
f= Prey(N,0.3,15)
prey =True
moth = not prey
Clock = pygame.time.Clock()
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type ==KEYDOWN:
            if event.key ==K_b:
                if screen_col == BLACK:
                    screen_col = WHITE
                else:
                    screen_col = BLACK
            if event.key ==K_p:
                old_pos =f.positions
                old_thetas = f.thetas
                f = Prey(N,0.3,15)
                f.positions = old_pos
                f.thetas = old_thetas
                prey =True
            

    # Fill the background with white
    screen.fill(screen_col)

    ## Update Predator Positions
    predator_pos = np.array(pygame.mouse.get_pos())
    pygame.draw.circle(screen,(255,0,0),predator_pos,10)

    # # Update Birds positions
    if prey:
        f.update_posdirs(1,0.1,predator_pos*15/500,repulsion_factor=5)
    elif moth:
        f.update_posdirs(1,1,predator_pos*15/500,attraction_factor=100)
    else:
        f.update_posdirs(1,0.1)
    p =np.floor(f.positions*500/15).astype("int32")
    for pos in p:
        pygame.draw.circle(screen,(0,0,255),pos,5)





    # Flip the display
    Clock.tick(40)
    pygame.display.flip()

pygame.quit()