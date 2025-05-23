#!/usr/bin/python
import pygame.locals as pgl
import pygame.gfxdraw
import pygame
from TBotTools import geometry, pgt, pid
import sys
import os
import numpy as np
import tensorflow as tf
import joblib
import torch
from snn_train import SNN  # Replace with the actual model class

currentpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(currentpath)
# Adjust the Python path to include the spkeras module
sys.path.append(os.path.join(currentpath, 'spkeras'))
clock = pygame.time.Clock()
dirpath = currentpath+'/Simulators/Images/'
framerate = 30  # set to 30 for Rasoberry pi
# -----------------------------------------------------------------------
#                           Physical constants
# -----------------------------------------------------------------------
dt = 1.0/framerate
sf = 1.0
acc_g = 9.81
l = 0.045  # distance between the centre of gravity of the T-Bot and the axil
R = 0.024  # Radius of wheels
C = 1     # Friction
h = l+R     # Maximum distance between the centre of gravity and the ground
t = 0
alpha = 0
gamma = 0
acc = 0
omega = 0
velocity = 0
distance = 0
theta = 0.05
height_of_man = 1.8923  # Me
# height_of_man = 0.1524 # 1:12 Scale (approx. 5-6") Action Figure
height_of_man = 0.0508  # 1:48 Scale (approx. 2") Action Figure
Tbot_scalefactor = 216
height_of_TBot_body = 120E-3
Man_scalefactor = (height_of_man/(l*2))*Tbot_scalefactor
wheel_radius = int(R/l*Tbot_scalefactor/2.2)
draw_stick_man = 1
tyre = 4
# -----------------------------------------------------------------------
#                           PID Cintrol
# -----------------------------------------------------------------------
targetvelocity = 0
speed_pid = pid.pid(0.040, 0.20, 0.00, [-10, 10], [-5, 5], dt)
angle_pid = pid.pid(12.00, 0.00, 0.20, [6, 6], [-1, 1], dt)
# -----------------------------------------------------------------------
#                          Drawing Geometry
# -----------------------------------------------------------------------
geom = geometry.geometry()
origin = [500, 319]
tbot_drawing_offset = [-78, -10]
tbot = np.loadtxt('T-BotSideView.dat')
# closes the shape and adds an offset
tbot = np.vstack((tbot, tbot[0, :]))+tbot_drawing_offset
tbot = tbot/(tbot[:, 1].max()-tbot[:, 1].min())*Tbot_scalefactor
spokes = np.array([[0, 1], [0, 0], [0.8660254, -0.5], [0, 0],
                  [-0.8660254, -0.5], [0, 0]])*(wheel_radius-tyre)
trackmarksArray = np.array(
    [[0, origin[1]+wheel_radius], [1000, origin[1]+wheel_radius]])
track_marks_tup = tuple(map(tuple, tuple((trackmarksArray).astype(int))))
# -----------------------------------------------------------------------
#                          Initialise Pygame
# -----------------------------------------------------------------------
pygame.init()
textPrint = pgt.TextPrint(pygame.Color('white'))
textPrint.setfontsize(18)
# Set the width and height of the screen (width, height).
screen = pygame.display.set_mode((1000, 700))
pygame.display.set_caption("T-Bot Simulator")
# Used to manage how fast the screen updates.
clock = pygame.time.Clock()
# -----------------------------------------------------------------------
#                           Load Images
# -----------------------------------------------------------------------
bg = pygame.image.load(dirpath+'/Gray.jpg').convert()
track_image = pygame.image.load(dirpath+'line.png')
# -----------------------------------------------------------------------
record = 0
framecount = 1
done = False

# Load the trained SNN model
snn_model = SNN()  # Replace with your model's initialization
snn_model.load_state_dict(torch.load('snn_model_weights.pth'))
snn_model.eval()  # Set the model to evaluation mode

# Load the scaler
scaler = joblib.load('scaler.pkl')

# ---------------------- Main Program Loop -----------------------------
while not done:
    g = acc_g * sf
    screen.blit(bg, (0, 0))
    screen.blit(track_image, (0, origin[1]+wheel_radius-8))
    # -------------------------------------------------------------------
    #                            The Physics
    # -------------------------------------------------------------------
    # if theta >= -np.pi/2.2 and theta <= np.pi/2.2:
    if -5 * np.pi <= theta <= 5 * np.pi:
        alpha = np.sin(theta) * g / h
        horizontal_acceleration = (alpha * R) + acc  # Horizontal acceleration from wheel rotation
        gamma = np.cos(theta) * horizontal_acceleration / h
        angular_acceleration = alpha - gamma

        # Integrate angular acceleration to get angular velocity
        omega += angular_acceleration * dt
        omega *= C  # Apply friction factor

        # Integrate angular velocity to get angle
        theta += omega * dt

        # Integrate dt to get time
        t += dt

        # Update velocity and distance
        velocity += acc * dt
        distance += velocity * dt

        # Add noise to the angle
        noise = np.random.rand(1) * np.pi / 180

        # Calculate target angle using the SNN model
        input_features = np.array([[theta, omega, velocity, targetvelocity, distance]])
        input_features = scaler.transform(input_features)  # Ensure features are scaled
        input_features = torch.tensor(input_features, dtype=torch.float32)  # Convert to PyTorch tensor
        acc = snn_model(input_features).item()  # Get the output from the SNN model

    # -------------------------------------------------------------------
    #                          Draw Stuff
    # -------------------------------------------------------------------
    if abs(theta) > np.pi/2:
        textPrint.abspos(screen, "Press the s key to reset.", (430, 580))
    mm2px = Tbot_scalefactor/height_of_TBot_body
    origin[0] = 500+int(distance*mm2px)+int(((theta)*np.pi)*wheel_radius/4)
    origin[0] = np.mod(origin[0], 1000)
    tbot_rot = np.array(geom.rotxy(theta+np.pi, tbot))
    tbot_tup = tuple(map(tuple, tuple((tbot_rot+origin).astype(int))))
    noise = np.random.rand(1)*np.pi/180
    spokes_rot = np.array(geom.rotxy(
        (distance*mm2px/wheel_radius)+theta, spokes))
    spokes_tup = tuple(map(tuple, tuple((spokes_rot+origin).astype(int))))
    pygame.gfxdraw.filled_polygon(screen, (tbot_tup), (0, 249, 249, 100))
    pygame.gfxdraw.aapolygon(screen, (tbot_tup), (255, 255, 255, 255))
    pygame.gfxdraw.aapolygon(screen, (spokes_tup), (255, 255, 255, 255))
    pygame.gfxdraw.aacircle(
        screen, origin[0], origin[1], wheel_radius-tyre, (255, 255, 255, 255))
    pygame.gfxdraw.aacircle(
        screen, origin[0], origin[1], wheel_radius, (255, 255, 255, 255))
    pygame.draw.lines(screen, (255, 255, 255, 255),
                      False, (track_marks_tup), 1)
    # -------------------------------------------------------------------
    #                          Get Key Pressed
    # -------------------------------------------------------------------
    for event in pygame.event.get():
        keys = pygame.key.get_pressed()
    if keys[pgl.K_s]:
        theta = 0.05
        omega = 0
        alpha = 0
        velocity = 0
    if keys[pgl.K_q]:
        done = True
    if keys[pygame.K_1]:
        record = 1
    if keys[pygame.K_2]:
        record = 0
        framecount = 1
    if record == 1:
        pygame.image.save(
            screen, "CapturedImages/{:04d}.png".format(framecount))
        framecount += 1
    if keys[pygame.K_p]:
        waiting = 1
        pressed = 1
        while waiting:
            for event in pygame.event.get():
                keys = pygame.key.get_pressed()
                if pressed:
                    textPrint.abspos(
                        screen, "Press o to return to simulator", (420, 500))
                pressed = 0
            if keys[pygame.K_o]:
                waiting = 0
            elif keys[pygame.K_q]:
                waiting = 0
                done = 1
            pygame.display.flip()
    pygame.display.flip()
    clock.tick(framerate)
pygame.display.quit()
pygame.quit()

print('Simulation Closed')
