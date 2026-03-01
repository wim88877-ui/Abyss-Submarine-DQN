import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import sys
from collections import deque

# ==========================================
# 1. Deep Q-Network (DQN) Architecture
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ==========================================
# 2. Game Entities Definition
# ==========================================
class Submarine:
    """Base class for Submarines"""
    def __init__(self, x, y, color):
        self.position = pygame.Vector2(x, y)
        self.color = color
        self.speed = 5
        self.max_hull = 10  # Hull integrity (Health points)
        self.hull = self.max_hull  
        self.size = 40
        self.facing = pygame.Vector2(0, -1) 

    def draw(self, screen):
        # Draw the main body of the submarine (ellipse)
        rect = (self.position.x, self.position.y, self.size, self.size)
        pygame.draw.ellipse(screen, self.color, rect)
        
        # Draw the sonar probe / front directional indicator
        center = (self.position.x + self.size/2, self.position.y + self.size/2)
        end_pos = (center[0] + self.facing.x * 25, center[1] + self.facing.y * 25)
        pygame.draw.line(screen, (200, 255, 255), center, end_pos, 3)

class Explorer(Submarine):
    """Player-controlled Explorer submarine"""
    def __init__(self, x, y):
        super().__init__(x, y, (0, 255, 150)) # Fluorescent sea green

    def handle_input(self, keys):
        if keys[pygame.K_w] or keys[pygame.K_UP]: 
            self.position.y -= self.speed
            self.facing = pygame.Vector2(0, -1)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]: 
            self.position.y += self.speed
            self.facing = pygame.Vector2(0, 1)
        if keys[pygame.K_a] or keys[pygame.K_LEFT]: 
            self.position.x -= self.speed
            self.facing = pygame.Vector2(-1, 0)
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: 
            self.position.x += self.speed
            self.facing = pygame.Vector2(1, 0)

class AUV(Submarine):
    """AI-controlled Autonomous Underwater Vehicle (RL Agent)"""
    def __init__(self, x, y):
        super().__init__(x, y, (255, 50, 100)) # Danger warning red
        self.action_space = [0, 1, 2, 3, 4] 
        self.facing = pygame.Vector2(0, 1)

    def choose_action(self, state, model, epsilon):
        if random.random() < epsilon:
            return random.choice(self.action_space) 
        else:
            state_tensor = state.unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
            return torch.argmax(q_values).item() 

    def perform_action(self, action):
        if action == 0: 
            self.position.y -= self.speed
            self.facing = pygame.Vector2(0, -1)
        elif action == 1: 
            self.position.y += self.speed
            self.facing = pygame.Vector2(0, 1)
        elif action == 2: 
            self.position.x -= self.speed
            self.facing = pygame.Vector2(-1, 0)
        elif action == 3: 
            self.position.x += self.speed
            self.facing = pygame.Vector2(1, 0)

class Torpedo:
    """Torpedo / Depth Charge obstacle"""
    def __init__(self, x, y, dx, dy, color=(255, 200, 0)):
        self.position = pygame.Vector2(x, y)
        self.direction = pygame.Vector2(dx, dy).normalize()
        self.speed = 8 # Simulate underwater drag
        self.color = color

    def update(self):
        self.position += self.direction * self.speed

    def draw(self, screen):
        # Draw the torpedo as a small capsule
        pygame.draw.circle(screen, self.color, (int(self.position.x), int(self.position.y)), 6)
        pygame.draw.circle(screen, (255,255,255), (int(self.position.x - self.direction.x*4), int(self.position.y - self.direction.y*4)), 3)

class Bubble:
    """Visual Effect: Rising deep sea bubbles"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.speed = random.uniform(1, 3)
        self.size = random.randint(2, 5)
        self.wobble = random.uniform(0, 2 * math.pi)

    def update(self):
        self.y -= self.speed
        self.x += math.sin(self.wobble) * 1.5
        self.wobble += 0.1

    def draw(self, screen):
        pygame.draw.circle(screen, (100, 150, 255, 128), (int(self.x), int(self.y)), self.size, 1)

# ==========================================
# 3. Trainer and Core RL Algorithm
# ==========================================
class Trainer:
    def __init__(self):
        self.model = DQN(42, 5) 
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001) 
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95 
        self.batch_size = 32 

    def get_state(self, auv, torpedoes):
        state = [auv.position.x, auv.position.y]
        for i in range(10):
            if i < len(torpedoes):
                t = torpedoes[i]
                state.extend([t.position.x, t.position.y, t.direction.x, t.direction.y])
            else:
                state.extend([0, 0, 0, 0])
        return torch.FloatTensor(state)

    def get_reward(self, auv, torpedoes, window_size):
        reward = 1.0 
        
        center = pygame.Vector2(window_size / 2, window_size / 2)
        dist_to_center = (auv.position - center).length()
        if dist_to_center > window_size * 2 / 5:
            reward -= 2.0

        for t in torpedoes:
            a = t.direction.y
            b = -t.direction.x
            c = t.direction.x * t.position.y - t.direction.y * t.position.x
            
            x1, y1 = auv.position.x, auv.position.y
            distance = abs(a * x1 + b * y1 + c) / ((a**2 + b**2)**0.5 + 1e-6)
            
            dx, dy = x1 - t.position.x, y1 - t.position.y
            in_front = (dx * t.direction.x + dy * t.direction.y) >= 0
            
            if in_front:
                if distance < auv.size: 
                    reward -= 2.0
                else:
                    safe_reward = min(distance, 200) / 200
                    reward += safe_reward * 0.1
        return reward

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q = self.model(states).gather(1, actions).squeeze(1)
        max_next_q = self.model(next_states).max(1)[0]
        target_q = rewards + (self.gamma * max_next_q * (1 - dones))

        criterion = nn.MSELoss()
        loss = criterion(current_q, target_q.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# ==========================================
# 4. Main Simulation Loop
# ==========================================
class AbyssSimulation:
    def __init__(self):
        pygame.init()
        self.window_size = 800
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Abyss Submarine - RL AUV Simulation")
        self.clock = pygame.time.Clock()
        
        self.font = pygame.font.Font(None, 28) 
        self.large_font = pygame.font.Font(None, 80) 
        self.title_font = pygame.font.Font(None, 100) 
        self.info_font = pygame.font.Font(None, 36)   
        
        # Deep sea color palette
        self.bg_color = (5, 15, 35) 
        
        self.trainer = Trainer()
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.bubbles = []
        self.ocean_current = -0.5 # Continuous leftward ocean current drift
        
        self.reset_env()

    def reset_env(self):
        self.explorer = Explorer(400, 700)
        self.auv = AUV(400, 100)
        self.torpedoes = []
        self.current_reward = 0.0
        self.last_action = "INITIALIZING"
        
    def show_start_screen(self):
        waiting = True
        # Pre-generate some ambient bubbles
        for _ in range(50):
            self.bubbles.append(Bubble(random.randint(0, self.window_size), random.randint(0, self.window_size)))
            
        while waiting:
            self.screen.fill(self.bg_color) 
            
            # Update ambient bubbles
            for b in self.bubbles:
                b.update()
                b.draw(self.screen)
                if b.y < 0:
                    b.y = self.window_size
                    b.x = random.randint(0, self.window_size)
            
            title_text = self.title_font.render("ABYSS SUBMARINE", True, (0, 255, 200))
            title_rect = title_text.get_rect(center=(self.window_size / 2, self.window_size / 3 - 50))
            self.screen.blit(title_text, title_rect)

            instructions = [
                "[ RL AUV SIMULATION TERMINAL ]",
                "W/A/S/D - Navigate Explorer",
                "SPACE - Launch Torpedo",
                "R - Reset Environment",
                "ESC - Terminate Protocol",
                "",
                ">> Press Y to Dive <<"
            ]
            
            start_y = self.window_size / 2
            for i, line in enumerate(instructions):
                color = (0, 255, 100) if ">>" in line else (150, 180, 200)
                text = self.info_font.render(line, True, color)
                rect = text.get_rect(center=(self.window_size / 2, start_y + i * 40))
                self.screen.blit(text, rect)

            pygame.display.flip()
            self.clock.tick(30)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_y:
                        waiting = False 
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

    def draw_ui(self):
        p_color = (0, 255, 150) if self.explorer.hull > 3 else (255, 50, 50)
        e_color = (255, 50, 100) if self.auv.hull > 3 else (255, 50, 50)
        player_text = self.font.render(f"Explorer Hull: {self.explorer.hull}/{self.explorer.max_hull}", True, p_color)
        auv_text = self.font.render(f"AUV Target Hull: {self.auv.hull}/{self.auv.max_hull}", True, e_color)
        
        rl_title = self.font.render("Neural Net Telemetry:", True, (255, 255, 255))
        eps_text = self.font.render(f"Exploration (Epsilon): {self.epsilon:.3f}", True, (200, 200, 200))
        rew_text = self.font.render(f"Q-Reward Signal: {self.current_reward:.1f}", True, (200, 200, 200))
        act_text = self.font.render(f"AUV Thruster: {self.last_action}", True, (200, 200, 200))
        env_text = self.font.render(f"Ocean Current: {self.ocean_current} m/s", True, (100, 150, 200))

        self.screen.blit(player_text, (20, self.window_size - 40))
        self.screen.blit(auv_text, (20, 20))
        
        self.screen.blit(rl_title, (self.window_size - 280, 20))
        self.screen.blit(eps_text, (self.window_size - 280, 50))
        self.screen.blit(rew_text, (self.window_size - 280, 80))
        self.screen.blit(act_text, (self.window_size - 280, 110))
        self.screen.blit(env_text, (self.window_size - 280, 140))

    def run(self):
        self.show_start_screen()
        running = True
        frame_count = 0
        action_names = ["NORTH", "SOUTH", "WEST", "EAST", "HOLD"]
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.torpedoes.append(Torpedo(self.explorer.position.x + 20, self.explorer.position.y, 0, -1, (0, 255, 255)))
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self.reset_env() 
            
            keys = pygame.key.get_pressed()
            self.explorer.handle_input(keys)

            if frame_count % 40 == 0:
                self.torpedoes.append(Torpedo(random.randint(0, self.window_size), 0, 0, 1, (255, 100, 100)))

            # Physics mechanic: Apply environmental ocean current thrust
            self.explorer.position.x += self.ocean_current
            self.auv.position.x += self.ocean_current

            current_state = self.trainer.get_state(self.auv, self.torpedoes)
            
            action = self.auv.choose_action(current_state, self.trainer.model, self.epsilon)
            self.auv.perform_action(action)
            self.last_action = action_names[action]

            # Boundary constraints
            self.explorer.position.x = max(0, min(self.explorer.position.x, self.window_size - self.explorer.size))
            self.explorer.position.y = max(0, min(self.explorer.position.y, self.window_size - self.explorer.size))
            self.auv.position.x = max(0, min(self.auv.position.x, self.window_size - self.auv.size))
            self.auv.position.y = max(0, min(self.auv.position.y, self.window_size - self.auv.size))

            reward = self.trainer.get_reward(self.auv, self.torpedoes, self.window_size)
            self.current_reward = reward
            done = False

            explorer_rect = pygame.Rect(self.explorer.position.x, self.explorer.position.y, self.explorer.size, self.explorer.size)
            auv_rect = pygame.Rect(self.auv.position.x, self.auv.position.y, self.auv.size, self.auv.size)

            for t in self.torpedoes[:]:
                t.position.x += self.ocean_current # Torpedoes are also affected by the current
                t.update()
                
                if t.position.y < 0 or t.position.y > self.window_size or t.position.x < 0 or t.position.x > self.window_size:
                    if t in self.torpedoes: self.torpedoes.remove(t)
                    continue
                
                if explorer_rect.collidepoint(t.position.x, t.position.y):
                    self.explorer.hull -= 1
                    if t in self.torpedoes: self.torpedoes.remove(t)
                    continue
                
                if auv_rect.collidepoint(t.position.x, t.position.y):
                    self.auv.hull -= 1
                    reward -= 50.0 
                    self.current_reward = reward
                    if t in self.torpedoes: self.torpedoes.remove(t)

            if self.explorer.hull <= 0 or self.auv.hull <= 0:
                done = True

            next_state = self.trainer.get_state(self.auv, self.torpedoes)
            self.trainer.memory.append((current_state, action, reward, next_state, done))
            self.trainer.train_step()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Visual rendering
            self.screen.fill(self.bg_color)
            
            # Render environmental bubbles
            if frame_count % 5 == 0:
                self.bubbles.append(Bubble(random.randint(0, self.window_size), self.window_size))
            for b in self.bubbles[:]:
                b.update()
                b.draw(self.screen)
                if b.y < 0:
                    self.bubbles.remove(b)
            
            self.explorer.draw(self.screen)
            self.auv.draw(self.screen)
            for t in self.torpedoes:
                t.draw(self.screen)
            
            self.draw_ui() 
            
            if done:
                res_text = "AUV NEUTRALIZED" if self.auv.hull <= 0 else "HULL BREACH DETECTED"
                res_color = (0, 255, 200) if self.auv.hull <= 0 else (255, 50, 50)
                game_over_text = self.large_font.render(res_text, True, res_color)
                text_rect = game_over_text.get_rect(center=(self.window_size/2, self.window_size/2))
                
                restart_hint = self.info_font.render("Press R to Restart Depth Simulation", True, (255, 255, 255))
                hint_rect = restart_hint.get_rect(center=(self.window_size/2, self.window_size/2 + 80))
                
                self.screen.blit(game_over_text, text_rect)
                self.screen.blit(restart_hint, hint_rect)
                pygame.display.flip()
                
                waiting_for_input = True
                while waiting_for_input:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            waiting_for_input = False
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_r:
                                self.reset_env()
                                waiting_for_input = False
                            elif event.key == pygame.K_ESCAPE:
                                running = False
                                waiting_for_input = False
            else:
                pygame.display.flip()
            
            self.clock.tick(60)
            frame_count += 1

        pygame.quit()

if __name__ == "__main__":
    simulation = AbyssSimulation()
    simulation.run()
