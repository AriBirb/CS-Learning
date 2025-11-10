import pygame
import random
pygame.font.init() # Initialize the font module

# For system fonts:
font = pygame.font.SysFont('Arial', 30)
font2 = pygame.font.SysFont('Arial', 60)
# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True
dt = 0

center = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)


class Game_Target:
    def __init__(self, radius: float, velocity: pygame.Vector2, position: pygame.Vector2, color: str):
        self.radius = radius
        self.velocity = velocity
        self.position = position
        self.color = color

    def update(self, dt):
        self.position += self.velocity * dt
        if self.position.x - self.radius < 0:
            self.velocity.x = -self.velocity.x
            self.position.x += -2 * (self.position.x - self.radius)
        elif self.position.x + self.radius > screen.get_width():
            self.velocity.x = -self.velocity.x
            self.position.x += 2 * (screen.get_width() - (self.position.x + self.radius))
        if self.position.y - self.radius < 0:
            self.velocity.y = -self.velocity.y
            self.position.y += -2 * (self.position.y - self.radius)
        elif self.position.y + self.radius > screen.get_height():
            self.velocity.y = -self.velocity.y
            self.position.y += 2 * (screen.get_height() - (self.position.y + self.radius))

    def draw(self):
        pygame.draw.circle(screen, self.color, self.position, self.radius)
        pygame.draw.circle(screen, 'white', self.position, self.radius * 0.666)
        pygame.draw.circle(screen, self.color, self.position, self.radius * 0.333)

    def detect_click(self, mouse_pos: pygame.Vector2):
        if ((mouse_pos.x - self.position.x) ** 2 + (mouse_pos.y - self.position.y) ** 2) ** 0.5 < self.radius:
            return True
        else:
            return False

def reset():
    global target, score, miss
    target = Game_Target(50, pygame.Vector2(0, 0), pygame.Vector2(screen.get_width()/2, screen.get_height()/2), "red")
    score = 0
    miss = ''

def game_over():
    screen.fill('black')
    end_text = font2.render("Game Over! Press SPACE to restart!", True, (255, 255, 255))
    score_text = font.render("Score: " + str(score), True, (255, 255, 255))
    miss_text = font2.render(miss, True, (255, 255, 255))
    screen.blit(end_text, (screen.get_width()/2 - 350, screen.get_height()/2-100))
    screen.blit(score_text, (30, 30))
    screen.blit(miss_text, (screen.get_width() - 200, 30))



target = Game_Target(50, pygame.Vector2(0,0), center, "red")
score = 0
miss = ''
game = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if game and event.type == pygame.MOUSEBUTTONDOWN:
            if target.detect_click(pygame.Vector2(event.pos)):
                # Target hit
                target.position.x = random.randint(int(target.radius), screen.get_width() - int(target.radius))
                target.position.y = random.randint(int(target.radius), screen.get_height() - int(target.radius))
                target.radius *= 0.975
                if target.velocity.length() == 0:
                    target.velocity = pygame.Vector2(30, 30)
                target.velocity = (target.velocity * 1.1).rotate(random.randint(0, 359))
                score += 1
            else:
                miss += 'X '
                if miss == 'X X X ':
                    game = False

        elif not game and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                game = True
                reset()

    # --- DRAWING AND UPDATING HAPPENS EVERY FRAME ---
    if game:
        screen.fill("black")
        target.draw()
        score_text = font.render("Score: " + str(score), True, (255, 255, 255))
        miss_text = font2.render(miss, True, (255, 255, 255))
        screen.blit(score_text, (30, 30))
        screen.blit(miss_text, (screen.get_width()-200, 30))
        target.update(dt)
    else:
        game_over()

    # Global escape key
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        pygame.quit()
        break

    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()
