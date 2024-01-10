import pygame,sys,torch
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel,self).__init__()
        self.fc=torch.nn.Linear(25,16)
        self.relu=torch.nn.ReLU()
        self.fc2=torch.nn.Linear(16,10)

    def forward(self,x):
        return self.fc2(self.relu(self.fc(x)))
torch.load("trained_model.pth",map_location=torch.device("cpu")).eval()
def convert_input_to_tensor(user_input):
    user_input_list=[[int(char) for char in row] for row in user_input]
    user_input_tensor=torch.tensor(user_input_list,dtype=torch.float32).view(1,-1)
    return user_input_tensor
pygame.init()
WIDTH,HEIGHT=650,650
GRID_SIZE,CELL_SIZE=5,WIDTH//GRID_SIZE
WHITE,BLACK,GRAY=(255,255,255),(0,0,0),(120,120,120)
grid=[[0]*GRID_SIZE for _ in range(GRID_SIZE)]
clicked_cells=[]
screen=pygame.display.set_mode((WIDTH,HEIGHT))
pygame.display.set_caption("Number AI")
font=pygame.font.Font(None,36)
output_border=pygame.Rect(5,5,WIDTH-10,50)
output_rect=pygame.Rect(10,10,WIDTH-20,40)
input_grid=[]
predicted_digit=""
while True:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type==pygame.MOUSEBUTTONDOWN:
            mouse_x,mouse_y=pygame.mouse.get_pos()
            grid_x,grid_y=mouse_x//CELL_SIZE,mouse_y//CELL_SIZE
            grid[grid_y][grid_x]=1-grid[grid_y][grid_x]
            if grid[grid_y][grid_x]==1:clicked_cells.append((grid_x,grid_y))
            else:clicked_cells.remove((grid_x,grid_y))
    screen.fill(WHITE)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            cell_color=BLACK if grid[y][x] == 1 else WHITE
            pygame.draw.rect(screen,cell_color,(x*CELL_SIZE,y*CELL_SIZE,CELL_SIZE,CELL_SIZE))
            pygame.draw.line(screen,BLACK,(x*CELL_SIZE,0),(x*CELL_SIZE,HEIGHT),3)
            pygame.draw.line(screen,BLACK,(0,y*CELL_SIZE),(WIDTH,y*CELL_SIZE),3)
    input_grid=[]
    for a,b,c,d,e in grid:
        input_grid.append(f"{a}{b}{c}{d}{e}")
        user_input_tensor=convert_input_to_tensor(input_grid)   
    if not len(clicked_cells) in [0,25]:
        with torch.no_grad():
            output=loaded_model(user_input_tensor)
        predicted_digit=str(torch.argmax(output).item())
    else:
        predicted_digit="..."
    pygame.draw.rect(screen,BLACK,output_border)
    pygame.draw.rect(screen,GRAY,output_rect)
    output_text=font.render(f"AI GUESS:{predicted_digit}",True,BLACK)
    screen.blit(output_text,(output_rect.x + 10,output_rect.y + 10))
    pygame.display.flip()
