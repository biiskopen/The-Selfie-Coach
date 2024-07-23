import pygame
print("starting")
pygame.init()
screen = pygame.display.set_mode([800, 600])
while True:

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            print("-> Exiting")
            pygame.quit()
            exit()
