import pygame
import random


# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

center = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)


class Player:
    def __init__(self, color: str, start_pos: pygame.Vector2, radius: int):
        self.color = color
        self.position = start_pos
        self.radius = radius

    def draw(self):
        pygame.draw.circle(screen, self.color, self.position, self.radius)

    def move_up(self):
        self.position.y -= 300 * dt

    def move_down(self):
        self.position.y += 300 * dt

    def move_left(self):
        self.position.x -= 300 * dt

    def move_right(self):
        self.position.x += 300 * dt

class Block:
    def __init__(self, color: str, start_pos: pygame.Vector2, width: int, height: int, speed: int):
        self.color = color
        self.position = start_pos
        self.width = width
        self.height = height
        self.rect = pygame.Rect(self.position.x, self.position.y, self.width, self.height)
        self.speed = speed

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect)

    def move_up(self):
        self.rect.y -= self.speed * dt

    def move_down(self):
        self.rect.y += self.speed * dt

    def move_left(self):
        self.rect.x -= self.speed * dt

    def move_right(self):
        self.rect.x += self.speed * dt

    def check_bottom_collision(self):
        if self.rect.y > screen.get_height():
            self.rect.y = 0-self.height
            self.rect.y += random.randint(-250, 0)
            self.rect.x += random.randint(-250, 250)
            self.rect.x = min(self.rect.x, screen.get_width()-self.width)
            self.rect.x = max(self.rect.x, 0)

    def stop_wall_collision(self):
        self.rect.x = min(self.rect.x, screen.get_width()-self.width)
        self.rect.x = max(0, self.rect.x)
        self.rect.y = min(self.rect.y, screen.get_height() - self.height)
        self.rect.y = max(0, self.rect.y)


    def check_two_block_colision(self, block):
        if (self.rect.x > block.rect.x and self.rect.x < block.rect.x + block.width and
                self.rect.y > block.rect.y and self.rect.y < block.rect.y + block.height):
            self.color = 'black'
        elif (self.rect.x + self.width > block.rect.x and self.rect.x + self.width < block.rect.x + block.width and
                self.rect.y > block.rect.y and self.rect.y < block.rect.y + block.height):
            self.color = 'black'
        elif (self.rect.x > block.rect.x and self.rect.x < block.rect.x + block.width and
                self.rect.y + self.height> block.rect.y and self.rect.y + self.height < block.rect.y + block.height):
            self.color = 'black'
        elif (self.rect.x + self.width > block.rect.x and self.rect.x + self.width < block.rect.x + block.width and
                self.rect.y + self.height > block.rect.y and self.rect.y + self.height < block.rect.y + block.height):
            self.color = 'black'
        elif (block.rect.x > self.rect.x and block.rect.x < self.rect.x + self.width and
                block.rect.y > self.rect.y and block.rect.y < self.rect.y + self.height):
            self.color = 'black'
        elif (block.rect.x + block.width > self.rect.x and block.rect.x + block.width < self.rect.x + self.width and
                block.rect.y > self.rect.y and block.rect.y < self.rect.y + self.height):
            self.color = 'black'
        elif (block.rect.x > self.rect.x and block.rect.x < self.rect.x + self.width and
                block.rect.y + block.height > self.rect.y and block.rect.y + block.height < self.rect.y + self.height):
            self.color = 'black'
        elif (block.rect.x + block.width > self.rect.x and block.rect.x + block.width < self.rect.x + self.width and
                block.rect.y + block.height > self.rect.y and block.rect.y + block.height < self.rect.y + self.height):
            self.color = 'black'



p1 = Block('red', center, 50, 50, 300)
b = []
for _ in range(45):
    b.append(Block('white', pygame.Vector2(random.randint(0, screen.get_width()-30), random.randint(-screen.get_width(), -60)), 30, 60, 200))

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")


    p1.draw()



    #pygame.draw.circle(screen, "red", player_pos, 40)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        p1.move_up()
    if keys[pygame.K_s]:
        p1.move_down()
    if keys[pygame.K_a]:
        p1.move_left()
    if keys[pygame.K_d]:
        p1.move_right()

    p1.stop_wall_collision()

    for block in b:
        block.draw()
        block.move_down()
        block.check_bottom_collision()
        block.speed += 12 * dt
        p1.check_two_block_colision(block)

    if keys[pygame.K_RSHIFT]:
        p1.color = 'red'
        p1.rect.x = int(center.x)
        p1.rect.y = int(center.y)
        for block in b:
            block.speed = 150
            block.rect.y = random.randint(-screen.get_height() -block.height, -block.height)
            block.rect.x = random.randint(0, screen.get_width() - block.width)

    if keys[pygame.K_ESCAPE]:
        pygame.quit()
        break

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    dt = clock.tick(60) / 1000

pygame.quit()
