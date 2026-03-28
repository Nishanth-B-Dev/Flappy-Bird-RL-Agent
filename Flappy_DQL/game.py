import pygame
import random

pygame.init()

WIDTH = 400
HEIGHT = 600

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flappy AI")

clock = pygame.time.Clock()


class Game:

    def __init__(self, render=True):

        self.render = render

        self.bird_y = 300
        self.velocity = 0

        self.pipe_x = 400
        self.pipe_gap = random.randint(200, 400)

        self.score = 0

        #  Score Font 
        self.font = pygame.font.SysFont("Arial", 30)

    def get_state(self):

        pipe_top = self.pipe_gap - 80
        pipe_bottom = self.pipe_gap + 80

        return [
            self.bird_y / HEIGHT,
            (self.pipe_x - 100) / WIDTH,
            (pipe_top - self.bird_y) / HEIGHT,
            (pipe_bottom - self.bird_y) / HEIGHT,
            self.velocity / 10
        ]

    def step(self, action):

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        if action == 1:
            self.velocity = -8

        self.velocity += 0.5
        self.bird_y += self.velocity

        self.pipe_x -= 4

        reward = 0.1
        done = False

        # Pipe boundaries
        pipe_top = self.pipe_gap - 80
        pipe_bottom = self.pipe_gap + 80

        #  Screen collision
        if self.bird_y < 0 or self.bird_y > HEIGHT:
            reward = -10
            done = True

        #  Pipe collision
        bird_x = 100
        bird_radius = 15
        pipe_width = 50

        if bird_x + bird_radius > self.pipe_x and bird_x - bird_radius < self.pipe_x + pipe_width:
            if self.bird_y - bird_radius < pipe_top or self.bird_y + bird_radius > pipe_bottom:
                reward = -10
                done = True

        #  Guidance reward
        if self.bird_y < pipe_top:
            reward -= 1
        elif self.bird_y > pipe_bottom:
            reward -= 1
        else:
            reward += 2

        #  Passed pipe
        if self.pipe_x < -50:
            self.pipe_x = WIDTH
            self.pipe_gap = random.randint(200, 400)
            self.score += 1
            reward += 5

        #  Render
        if self.render:
            screen.fill((135, 206, 235))

            pygame.draw.circle(screen, (255, 255, 0), (100, int(self.bird_y)), 15)

            pygame.draw.rect(screen, (0, 255, 0), (self.pipe_x, 0, 50, pipe_top))
            pygame.draw.rect(screen, (0, 255, 0), (self.pipe_x, pipe_bottom, 50, HEIGHT))

            #  DRAW SCORE
            score_text = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
            screen.blit(score_text, (10, 10))

            pygame.display.update()
            clock.tick(60)
        else:
            clock.tick(0)

        return self.get_state(), reward, done