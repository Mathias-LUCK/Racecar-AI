import sys, math, time, pygame, tkinter as tk, tkinter.messagebox as mb
import os
from typing import List, Optional
import numpy as np, pygame.gfxdraw
from pygame.math import Vector2
from reward_system import RewardSystem

os.chdir(os.path.dirname(os.path.abspath(__file__)))

reward_system = RewardSystem()

pygame.init()
W, H = 1200, 800
BG_GREEN  = (16, 128,  84)
GREY_ROAD = (140, 140, 140)
BLUE_RAY  = (  0, 120, 255)
WHITE     = (255, 255, 255)
TICK_SPEED = 165
timeout_duration = 1

total_episode_reward = 0
last_death_moment = time.time()

# (accelerate, brake, left, right) — 8 discrete actions
Actions_list = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1),
                (0, 0, 1, 0), (0, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, 1)]

screen    = pygame.display.set_mode((W, H))
pygame.display.set_caption("Racecar AI")
clock     = pygame.time.Clock()
road_surf = pygame.Surface((W, H))


def seg_intersect(a, b, c, d) -> bool:
    orient = lambda p, q, r: (q.x-p.x)*(r.y-p.y) - (q.y-p.y)*(r.x-p.x)
    A, B, C, D = map(Vector2, (a, b, c, d))
    o1, o2, o3, o4 = orient(A,B,C), orient(A,B,D), orient(C,D,A), orient(C,D,B)
    if o1 == o2 == o3 == o4 == 0:
        return False
    return (o1*o2 <= 0) and (o3*o4 <= 0)

def dist_point_seg(p: Vector2, a: Vector2, b: Vector2) -> float:
    if a == b:
        return (p-a).length()
    t = max(0, min(1, (p-a).dot(b-a)/(b-a).length_squared()))
    return (p - (a + t*(b-a))).length()

def raycast(origin: Vector2, ang: float, max_len=600, step=2) -> float:
    d = Vector2(math.cos(ang), math.sin(ang))
    for dist in range(0, max_len, step):
        x, y = int(origin.x + d.x*dist), int(origin.y + d.y*dist)
        if not (0 <= x < W and 0 <= y < H) or road_surf.get_at((x,y)) == BG_GREEN:
            return dist
    return max_len

# 5 raycasts: hard left, soft left, forward, soft right, hard right
RAY_OFFSETS = [-math.radians(60), -math.radians(30), 0,
                math.radians(30),  math.radians(60)]


class AI:
    def __init__(self):
        self.obs = []
        self.best_weights = None
        self.current_weights = [np.random.uniform(-5, 5, (6, 8)), np.random.uniform(-5, 5, 8)]
        self.max_episode_reward = -np.inf
        self.generation = 1
        self.prob_distribution = []
        self.load_best_weights()

    def save_best_weights(self):
        if self.best_weights is not None:
            weights_data = {
                'weights_matrix': self.best_weights[0].tolist(),
                'bias_vector': self.best_weights[1].tolist(),
                'max_reward': self.max_episode_reward,
                'generation': self.generation
            }
            with open('Best_weights.py', 'w') as f:
                f.write(f"# Best AI weights - Max reward: {self.max_episode_reward:.2f}, Generation: {self.generation}\n")
                f.write(f"best_weights_data = {repr(weights_data)}\n")
            print(f"💾 Saved best weights! Reward: {self.max_episode_reward:.2f}, Gen: {self.generation}")

    def load_best_weights(self):
        try:
            if os.path.exists('Best_weights.py'):
                import importlib.util
                spec = importlib.util.spec_from_file_location("best_weights", "Best_weights.py")
                best_weights_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(best_weights_module)
                data = best_weights_module.best_weights_data
                self.best_weights = [
                    np.array(data['weights_matrix']),
                    np.array(data['bias_vector'])
                ]
                self.current_weights = [w.copy() for w in self.best_weights]
                self.max_episode_reward = data['max_reward']
                self.generation = data.get('generation', 1)
                print(f"🚀 Loaded best weights! Reward: {self.max_episode_reward:.2f}, Gen: {self.generation}")
            else:
                print("📝 No existing weights found, starting fresh!")
        except Exception as e:
            print(f"⚠️ Error loading weights: {e}")
            print("Starting with random weights...")

    def reset_weights(self):
        self.best_weights = None
        self.current_weights = [np.random.uniform(-5, 5, (6, 8)), np.random.uniform(-5, 5, 8)]
        self.max_episode_reward = -np.inf
        self.generation = 1
        self.prob_distribution = []
        if os.path.exists('Best_weights.py'):
            os.remove('Best_weights.py')
        print("🔄 Weights reset to random")

    def get_action(self):
        # (6,) × (6,8) + (8,) = (8,) → softmax
        action_values = np.dot(self.obs, self.current_weights[0]) + self.current_weights[1]
        ai.prob_distribution = np.exp(action_values) / np.sum(np.exp(action_values))
        return np.argmax(ai.prob_distribution)

    def generate_next_generation(self):
        if self.best_weights is None:
            return self.current_weights
        new_gen_weights = [w.copy() for w in self.best_weights]
        # mutation factor decreases over time so the search narrows in
        mutation_factor = (-self.generation/10000 + 1.5)
        new_gen_weights[0] = (new_gen_weights[0] * np.random.uniform(0.65, 1.35, new_gen_weights[0].shape) + np.random.uniform(-0.15, 0.05, new_gen_weights[0].shape)) * mutation_factor
        new_gen_weights[1] = (new_gen_weights[1] * np.random.uniform(0.65, 1.35, new_gen_weights[1].shape) + np.random.uniform(-0.15, 0.05, new_gen_weights[1].shape)) * mutation_factor
        self.generation += 1
        # keep weights from exploding
        if max(np.max(np.abs(self.best_weights[0])), np.max(np.abs(self.best_weights[1]))) > 10:
            self.best_weights[0] /= 1.6
            self.best_weights[1] /= 1.6
        print(f"Generation: {self.generation}")
        print(f"Current best reward: {self.max_episode_reward}")
        return new_gen_weights


class Car:
    def __init__(self):
        self.base_img = pygame.image.load("car.png").convert_alpha()
        self.scale = 1.0
        self._update_master()
        self.x, self.y, self.angle = W//2, H//2, 0.0
        self.vel = 0.0
        self.set_difficulty(1)
        self.update_rect()

    def _update_master(self):
        w, h = 30, 55
        img = pygame.transform.scale(self.base_img, (int(w*self.scale), int(h*self.scale)))
        self.img_master = pygame.transform.rotate(img, -90)

    def update_rect(self):
        self.img  = pygame.transform.rotate(self.img_master, math.degrees(-self.angle))
        self.rect = self.img.get_rect(center=(self.x, self.y))

    def _corners(self):
        w, h = self.img_master.get_size()
        local = [Vector2(-w/2,-h/2), Vector2(w/2,-h/2),
                 Vector2(w/2, h/2),  Vector2(-w/2,h/2)]
        fwd  = Vector2(math.cos(self.angle), math.sin(self.angle))
        left = fwd.rotate(90)
        return [Vector2(self.x, self.y)+v.x*fwd+v.y*left for v in local]

    def set_scale(self, s: float):
        self.scale = s; self._update_master(); self.update_rect()

    def set_difficulty(self, d: int):
        base = {0:4.5,1:9.0,2:11.5}[d]*1.30
        self.fwd, self.bwd = 0.85*base, -0.35*base
        self.turn_c = {0:11,1:9.5,2:8}[d]*(np.pi/2000)

    def tick(self, action):
        steer = -1 if (action == 2 or action == 4 or action == 6) else 1 if (action == 3 or action == 5 or action == 7) else 0
        if action in [0, 4, 5]:
            self.vel -= self.vel*0.0045
            if abs(self.vel)<0.01: self.vel=0
        if action in [1, 2, 3]:
            self.vel = min(self.vel + (self.fwd-1.3*self.vel)*0.0035, self.fwd)
        if action in [5, 6]:
            self.vel = max(self.vel + (self.bwd-1.3*self.vel)*0.008, self.bwd)
        if abs(self.vel)>0.01 and steer:
            rate=min(np.pi/200,(self.turn_c/1.25)*abs(self.vel))
            self.angle += rate * (-1 if (steer==-1) ^ (self.vel<0) else 1)
            self.vel   *= 0.99875
        self.x += self.vel*math.cos(self.angle)
        self.y += self.vel*math.sin(self.angle)
        self.update_rect()

    def tick_manual(self, keys):
        steer = 0
        if keys[pygame.K_LEFT]:  steer = -1
        if keys[pygame.K_RIGHT]: steer =  1
        accel = keys[pygame.K_UP]
        brake = keys[pygame.K_DOWN]

        if not (accel or brake):
            self.vel -= self.vel * 0.0045
            if abs(self.vel) < 0.01: self.vel = 0
        if accel:
            self.vel = min(self.vel + (self.fwd - 1.3*self.vel)*0.0035, self.fwd)
        if brake:
            self.vel = max(self.vel + (self.bwd - 1.3*self.vel)*0.008, self.bwd)
        if abs(self.vel) > 0.01 and steer:
            rate = min(np.pi/200, (self.turn_c/1.25)*abs(self.vel))
            self.angle += rate * (-1 if (steer == -1) ^ (self.vel < 0) else 1)
            self.vel   *= 0.99875
        self.x += self.vel * math.cos(self.angle)
        self.y += self.vel * math.sin(self.angle)
        self.update_rect()

    def draw(self,surf): surf.blit(self.img,self.rect)


class Track:
    def __init__(self, pts=None, width=80, checkpoints=None, finish=None):
        self.pts=list(pts) if pts else []; self.width=width
        self.checkpoints=list(checkpoints) if checkpoints else []; self.finish=finish
    def add_pt(self,pos):
        if not self.pts or Vector2(pos).distance_to(self.pts[-1])>=8: self.pts.append(pos)
    def set_width(self,w): self.width=w
    def draw(self,surf,show_cp=True,passed=None):
        surf.fill(BG_GREEN); n=len(self.pts)
        if n>=2:
            for i in range(n):
                a,b=Vector2(self.pts[i]),Vector2(self.pts[(i+1)%n]); v=b-a
                if v.length_squared()<1: continue
                perp=v.normalize().rotate(90)*(self.width/2)
                quad=[a+perp,a-perp,b-perp,b+perp]
                pygame.gfxdraw.filled_polygon(surf,[(int(p.x),int(p.y)) for p in quad],GREY_ROAD)
                pygame.gfxdraw.filled_circle(surf,int(a.x),int(a.y),self.width//2,GREY_ROAD)
            pygame.gfxdraw.filled_circle(surf,int(self.pts[-1][0]),int(self.pts[-1][1]),
                                         self.width//2,GREY_ROAD)
        if show_cp:
            off,on=(220,40,40),(40,220,40)
            for i,(p,q) in enumerate(self.checkpoints):
                pygame.draw.line(surf,on if passed and passed[i] else off,p,q,4)
            if self.finish: pygame.draw.line(surf,(240,240,0),*self.finish,6)


def rebuild_road(): track.draw(road_surf,False)
def off_road():
    return any(0<=p.x<W and 0<=p.y<H and road_surf.get_at((int(p.x),int(p.y)))==BG_GREEN
               for p in car._corners())
def respawn():
    global cp_passed, lap_started, last_death_moment, total_episode_reward
    car.x,car.y,car.angle=spawn; car.vel=0; car.update_rect()
    cp_passed=[False]*len(track.checkpoints); lap_started=False
    reward_system.reset_episode()
    last_death_moment = time.time()
    print(total_episode_reward)
    total_episode_reward = 0
def speed_kmh(): return abs(round(car.vel*500/7.65*TICK_SPEED/120))
def fmt(t): return f"{t:.2f}" if t is not None else "–"


try: from tracks import tracks as saved
except ImportError: saved=[]
def save_current():
    entry=dict(pts=track.pts,width=track.width,car_scale=car.scale,
               spawn=spawn if spawn_set else None,
               checkpoints=track.checkpoints,finish=track.finish)
    idx=cur_idx if cur_idx!=-1 else len(saved)
    if cur_idx==-1: saved.append(entry)
    else: saved[idx]=entry
    with open("tracks.py","w") as f: f.write("tracks = "+repr(saved))
    refresh_list(idx)
def load(idx:Optional[int]):
    global cur_idx,track,spawn,spawn_set,cp_passed,lap_started,lap_times,lap_start_time
    global car_following_mouse
    if idx is None:
        cur_idx=-1; track=Track(); spawn_set=False
        cp_passed=[]; lap_times=[]; lap_started=False; lap_start_time=0
        car.x,car.y,car.angle=W//2,H//2,0; car.update_rect()
    else:
        cur_idx=idx; d=saved[idx]
        track=Track(d["pts"],d["width"],d.get("checkpoints"),d.get("finish"))
        spawn=d.get("spawn",(W//2,H//2,0)); spawn_set=d.get("spawn") is not None
        car.set_scale(d.get("car_scale",1.0))
        car.x,car.y,car.angle=spawn; car.update_rect()
        cp_passed=[False]*len(track.checkpoints)
        lap_times=[]; lap_started=False; lap_start_time=0
    car_following_mouse=False
    width_slider.set(track.width); car_slider.set(int(car.scale*100))
    rebuild_road()

def do_reset_weights():
    if mb.askyesno("Reset weights", "Delete saved weights and restart from random?"):
        ai.reset_weights()
        respawn()

root=tk.Tk(); root.title("Manager"); root.geometry("+50+50")
track_var=tk.StringVar(); mode_var=tk.StringVar(value="Editor")
sub_var=tk.StringVar(value="Track"); place_var=tk.StringVar(value="Checkpoint")

width_slider=tk.Scale(root,from_=40,to=160,orient="horizontal",label="Road width",
                      command=lambda v:(track.set_width(int(v)),rebuild_road()))
car_slider=tk.Scale(root,from_=50,to=200,orient="horizontal",label="Car size %",
                    command=lambda v:car.set_scale(int(v)/100))
def refresh_list(sel=None):
    opts=["New Track"]+[f"Track {i}" for i in range(len(saved))]
    menu["menu"].delete(0,"end")
    for i,lbl in enumerate(opts):
        menu["menu"].add_command(label=lbl,
            command=lambda i=i:(track_var.set(opts[i]),load(None if i==0 else i-1)))
    track_var.set(opts[sel+1] if sel is not None else opts[0])
menu=tk.OptionMenu(root,track_var,()); menu.pack(padx=8,pady=(8,4),fill="x")
refresh_list()
tk.OptionMenu(root,mode_var,"Editor","AI","Drive").pack(padx=8,pady=4,fill="x")
tk.Label(root,text="Editor mode:").pack(anchor="w",padx=8)
for lbl in ("Track","Car","Checkpoint"):
    tk.Radiobutton(root,text=lbl,variable=sub_var,value=lbl).pack(anchor="w")
cp_frame=tk.Frame(root); cp_frame.pack(anchor="w",padx=20)
tk.Radiobutton(cp_frame,text="Place Checkpoint",variable=place_var,
               value="Checkpoint").pack(anchor="w")
tk.Radiobutton(cp_frame,text="Place Finish Line",variable=place_var,
               value="Finish").pack(anchor="w")
width_slider.pack(fill="x",padx=8,pady=(6,2))
car_slider.pack(fill="x",padx=8,pady=(0,6))
tk.Button(root,text="SAVE",command=save_current).pack(fill="x",padx=8,pady=4)
tk.Button(root,text="Reset weights",command=do_reset_weights).pack(fill="x",padx=8,pady=(0,8))
root.protocol("WM_DELETE_WINDOW",lambda:None)

car=Car(); track=Track(); ai=AI(); reward_system=RewardSystem(); spawn=(W//2,H//2,0); spawn_set=False; cur_idx=-1
cp_passed:List[bool]=[]; lap_times:List[float]=[]
lap_started=False; lap_start_time=0.0
pending_pt=None; car_following_mouse=False
rebuild_road()

try: load(4); mode_var.set("AI")
except IndexError: print("⚠️  Track 5 not found – starting blank")

font_speed=pygame.font.SysFont("Arial",30)
font_hud  =pygame.font.SysFont("Arial",24)

running=True
while running:
    for e in pygame.event.get():
        if e.type==pygame.QUIT: running=False
        if mode_var.get()=="Editor":
            if sub_var.get()=="Car":
                if e.type==pygame.MOUSEWHEEL and car_following_mouse:
                    car.angle+=e.y*0.05; car.update_rect()
                if e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
                    car_following_mouse=not car_following_mouse
                    if not car_following_mouse:
                        spawn=(car.x,car.y,car.angle); spawn_set=True
            if sub_var.get()=="Checkpoint" and e.type==pygame.MOUSEBUTTONDOWN and e.button==1:
                if pending_pt is None: pending_pt=e.pos
                else:
                    if place_var.get()=="Checkpoint":
                        track.checkpoints.append((pending_pt,e.pos)); cp_passed.append(False)
                    else: track.finish=(pending_pt,e.pos)
                    pending_pt=None; rebuild_road()
        if mode_var.get()=="Drive" and e.type==pygame.KEYDOWN and e.key==pygame.K_r:
            respawn()

    keys=pygame.key.get_pressed()
    mode=mode_var.get()

    if mode=="Editor":
        if sub_var.get()=="Track" and pygame.mouse.get_pressed()[0]:
            track.add_pt(pygame.mouse.get_pos()); rebuild_road()
        elif sub_var.get()=="Car" and car_following_mouse:
            car.x,car.y=pygame.mouse.get_pos(); car.update_rect()

    elif mode=="AI":
        if not spawn_set:
            mb.showwarning("Spawn missing","Define a spawn point first!")
            mode_var.set("Editor")
        else:
            origin=Vector2(car.x,car.y)
            rays=[raycast(origin,car.angle+o) for o in RAY_OFFSETS]
            # normalize each ray by its max useful range at that angle
            rays[0]/=140; rays[1]/=375; rays[2]/=550; rays[3]/=375; rays[4]/=140
            try:
                idx=cp_passed.index(False); a,b=map(Vector2,track.checkpoints[idx])
                dist_cp=dist_point_seg(origin,a,b)
            except ValueError: dist_cp=float('inf')
            prev=car._corners()
            ai.obs=[round(car.vel,3)]+[round(r,1) for r in rays]
            action=ai.get_action()
            car.tick(action)
            step_reward=reward_system.calculate_reward(car,track,cp_passed,ai.prob_distribution)
            total_episode_reward+=step_reward

            time_since__last_death=time.time()-last_death_moment

            if not lap_started and abs(car.vel)>0.05:
                lap_started=True; lap_start_time=time.time()

            if off_road() or time.time()-reward_system.last_checkpoint_time>timeout_duration:
                death_penalty=reward_system.calculate_reward(car,track,cp_passed,prob_distribution=[0]*8,died=True)
                total_episode_reward+=death_penalty
                if total_episode_reward>ai.max_episode_reward:
                    ai.best_weights=ai.current_weights.copy()
                    ai.save_best_weights()
                    ai.max_episode_reward=total_episode_reward
                ai.current_weights=ai.generate_next_generation()
                respawn(); reward_system.reset_after_death()

            elif time_since__last_death>50:
                step_reward=reward_system.calculate_reward(car,track,cp_passed,ai.prob_distribution)
                total_episode_reward+=step_reward
                if total_episode_reward>ai.max_episode_reward:
                    ai.best_weights=ai.current_weights.copy()
                    ai.max_episode_reward=total_episode_reward
                    ai.save_best_weights()
                ai.current_weights=ai.generate_next_generation()
                respawn(); reward_system.reset_after_death()

            else:
                cur=car._corners()
                for i,(p,q) in enumerate(track.checkpoints):
                    if not cp_passed[i] and any(seg_intersect(pc,cc,p,q) for pc,cc in zip(prev,cur)):
                        cp_passed[i]=True; reward_system.checkpoint_passed()
                if track.finish and all(cp_passed):
                    if any(seg_intersect(pc,cc,*track.finish) for pc,cc in zip(prev,cur)):
                        now=time.time()
                        if lap_started:
                            lap_time=now-lap_start_time
                            lap_times.insert(0,lap_time); lap_times=lap_times[:3]
                            lap_reward=reward_system.calculate_reward(car,track,cp_passed,lap_time=lap_time)
                            total_episode_reward+=lap_reward
                        lap_start_time=now; lap_started=True
                        cp_passed=[False]*len(track.checkpoints)

    elif mode=="Drive":
        if not spawn_set:
            mb.showwarning("Spawn missing","Define a spawn point first!")
            mode_var.set("Editor")
        else:
            prev=car._corners()
            car.tick_manual(keys)

            if not lap_started and abs(car.vel)>0.05:
                lap_started=True; lap_start_time=time.time()

            if off_road():
                respawn()
            else:
                cur=car._corners()
                for i,(p,q) in enumerate(track.checkpoints):
                    if not cp_passed[i] and any(seg_intersect(pc,cc,p,q) for pc,cc in zip(prev,cur)):
                        cp_passed[i]=True
                if track.finish and all(cp_passed):
                    if any(seg_intersect(pc,cc,*track.finish) for pc,cc in zip(prev,cur)):
                        now=time.time()
                        if lap_started:
                            lap_time=now-lap_start_time
                            lap_times.insert(0,lap_time); lap_times=lap_times[:3]
                        lap_start_time=now; lap_started=True
                        cp_passed=[False]*len(track.checkpoints)

    now=time.time()

    screen.blit(road_surf,(0,0))
    track.draw(screen,True,cp_passed)
    if (mode_var.get()=="Editor" and sub_var.get()=="Car") or mode in ("AI","Drive"):
        car.draw(screen)
    if mode=="AI":
        origin=Vector2(car.x,car.y)
        for o in RAY_OFFSETS:
            d=raycast(origin,car.angle+o)
            end=origin+Vector2(math.cos(car.angle+o),math.sin(car.angle+o))*d
            pygame.draw.line(screen,BLUE_RAY,(int(origin.x),int(origin.y)),(int(end.x),int(end.y)),2)

    sp=font_speed.render(str(speed_kmh()),True,WHITE)
    screen.blit(sp,(W-sp.get_width()-8,8))

    live=(now-lap_start_time) if lap_started else None
    last=lap_times[0] if lap_times else None
    pb=min(lap_times) if lap_times else None
    txt=[f"Lap : {fmt(live)}" if live else "Lap : –",
         f"Last: {fmt(last)}",
         f" PB : {fmt(pb)}"]
    y=8
    for line in txt:
        surf=font_hud.render(line,True,WHITE)
        screen.blit(surf,(8,y)); y+=surf.get_height()+2

    if mode=="Drive":
        hint=font_hud.render("R = respawn",True,(180,180,180))
        screen.blit(hint,(8, H-hint.get_height()-8))

    pygame.display.flip()
    clock.tick(TICK_SPEED)
    root.update_idletasks(); root.update()

pygame.quit(); root.destroy(); sys.exit()
