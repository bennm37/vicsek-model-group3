import pygame
from flock import *

#Simple pygame program


# Import and initialize the pygame library
pygame.init()


# Set up the drawing window

screen = pygame.display.set_mode([500, 500])

# Run until the user asks to quit
running = True

##Setting up the Flock
f= Flock(10,0.5,15)
while running:

    # Did the user click the window close button?
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background with white
    screen.fill((255, 255, 255))

    # Update Birds positions
    f.update_posdirs(1,0.1)
    p =np.floor(f.positions*500/15,dtype=int)
    for pos in p:
        pygame.draw.circle(screen,(0,0,255),pos,25)

    pygame.draw.circle(screen, (0, 0, 255), (250, 250), 25)


    # Flip the display
    pygame.display.flip()

pygame.quit()