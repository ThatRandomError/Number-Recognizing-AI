import pygame as py
import sys
import torch

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = torch.nn.Linear(25, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 10)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

loaded_model = torch.load('trained_model.pth', map_location=torch.device('cpu'))
loaded_model.eval()

def convert_input_to_tensor(user_input):
    user_input_list = [[int(char) for char in row] for row in user_input]
    user_input_tensor = torch.tensor(user_input_list, dtype=torch.float32).view(1, -1)
    return user_input_tensor
py.init()

WIDTH, HEIGHT = 650,650
GRID_SIZE = 5
CELL_SIZE = WIDTH//GRID_SIZE
WHITE = (255,255,255)
BLACK = (0,0,0)
GRAY = (120,120,120)

grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]

clicked_cells = []

screen = py.display.set_mode((WIDTH, HEIGHT))
py.display.set_caption("Number AI")

font = py.font.Font(None, 36)

output_border = py.Rect(5, 5, WIDTH-10, 50)
output_rect = py.Rect(10, 10, WIDTH-20, 40)

input_grid = []

predicted_digit = ''

while True:
    for event in py.event.get():
        if event.type == py.QUIT:
            py.quit()
            sys.exit()
        elif event.type == py.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = py.mouse.get_pos()
            
            grid_x = mouse_x//CELL_SIZE
            grid_y = mouse_y//CELL_SIZE

            grid[grid_y][grid_x] = 1 - grid[grid_y][grid_x]
            if grid[grid_y][grid_x] == 1:
                clicked_cells.append((grid_x,grid_y))
            else:
                clicked_cells.remove((grid_x,grid_y))

    screen.fill(WHITE)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            cell_color = BLACK if grid[y][x] == 1 else WHITE
            py.draw.rect(screen, cell_color, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            py.draw.line(screen, BLACK, (x * CELL_SIZE, 0), (x * CELL_SIZE, HEIGHT), 3)
            py.draw.line(screen, BLACK, (0, y * CELL_SIZE), (WIDTH, y * CELL_SIZE), 3)

    input_grid = []
    for a, b, c, d, e in grid:
        input_grid.append(f"{a}{b}{c}{d}{e}")
        user_input_tensor = convert_input_to_tensor(input_grid)   

    if(len(clicked_cells) != 0):
        with torch.no_grad():
            output = loaded_model(user_input_tensor)
        predicted_digit = str(torch.argmax(output).item())
    else:
        predicted_digit = "..."

    py.draw.rect(screen, BLACK, output_border)
    py.draw.rect(screen, GRAY, output_rect)

    output_text = font.render(f"AI GUESS: {predicted_digit}", True, BLACK)
    screen.blit(output_text, (output_rect.x + 10, output_rect.y + 10))

    py.display.flip()

