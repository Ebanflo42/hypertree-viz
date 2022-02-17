import numpy.random as rd
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

vertices = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, 1, 1),
    (-1, -1, 1)
)

colors = [(rd.uniform(), rd.uniform(), rd.uniform()) for _ in vertices]

edges = (
    (0, 1),
    (0, 3),
    (0, 4),
    (2, 1),
    (2, 3),
    (2, 6),
    (6, 7),
    (5, 1),
    (5, 4),
    (5, 6),
    (7, 3),
    (7, 4)
)

surfaces = (
    (0, 1, 2, 3),
    (3, 2, 7, 6),
    (6, 7, 5, 4),
    (4, 5, 1, 0),
    (1, 5, 7, 2),
    (4, 0, 3, 6)
)

def Cube():

    """
    glBegin(GL_QUADS)
    for surface in surfaces:
        glColor3fv(())
    """
    #"""
    glBegin(GL_POINTS)
    for i, vertex in enumerate(vertices):
        glVertex3fv(vertex)
        glColor3fv(colors[i])
    glEnd()
    glBegin(GL_LINES)
    glColor3fv((1, 1, 1))
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
            glColor3fv((1, 1, 1))
    glEnd()
    #"""

def main():

    pygame.init()
    display = (1920, 1080)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, display[0]/display[1], 0.1, 50.0)
    #glOrtho(-display[0]//2, display[0]//2, display[1]//2, display[1]//2, -1, 10)

    glTranslatef(0.0, 0.0, -5.0)

    glRotatef(0, 0, 0, 0)

    glPointSize(8)

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 3, 1, 1)

        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pygame.display.flip()
        pygame.time.wait(10)

main()

