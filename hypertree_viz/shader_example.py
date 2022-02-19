"""Example rotating a point cloud containing 100.000 points"""
from __future__ import print_function, division

from OpenGL import GL, GLU
import OpenGL.GL.shaders
import ctypes
import pyautogui
import numpy as np
import pyopengltk
import sys
import time
from tqdm import tqdm
import pickle as pkl

if sys.version_info[0] > 2:
    import tkinter as tk
else:
    import Tkinter as tk


# Avoiding glitches in pyopengl-3.0.x and python3.4
def bytestr(s):
    return s.encode("utf-8") + b"\000"


# Avoiding glitches in pyopengl-3.0.x and python3.4
def compileShader(source, shaderType):
    """
    Compile shader source of given type
        source -- GLSL source-code for the shader
    shaderType -- GLenum GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, etc,
        returns GLuint compiled shader reference
    raises RuntimeError when a compilation failure occurs
    """
    if isinstance(source, str):
        source = [source]
    elif isinstance(source, bytes):
        source = [source.decode('utf-8')]

    shader = GL.glCreateShader(shaderType)
    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)
    result = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not(result):
        # TODO: this will be wrong if the user has
        # disabled traditional unpacking array support.
        raise RuntimeError(
            """Shader compile failure (%s): %s""" % (
                result,
                GL.glGetShaderInfoLog(shader),
            ),
            source,
            shaderType,
        )
    return shader


with open('hypertree_viz/vert.glsl', 'r') as f:
    vertex_shader = f.read()

with open('hypertree_viz/frag.glsl', 'r') as f:
    fragment_shader = f.read()

vertices = np.load('vertices.npy')
colors = np.load('colors.npy')
NPTS = len(vertices)


def create_object(shader):
    # Create a new VAO (Vertex Array Object) and bind it
    vertex_array_object = GL.glGenVertexArrays(1)
    GL.glBindVertexArray(vertex_array_object)

    # Generate buffers to hold our vertices
    vertex_buffer = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vertex_buffer)

    # Get the position of the 'position' in parameter of our shader
    # and bind it.
    position = GL.glGetAttribLocation(shader, bytestr('position'))
    GL.glEnableVertexAttribArray(position)

    # Describe the position data layout in the buffer
    GL.glVertexAttribPointer(position, 3, GL.GL_FLOAT, False,
                             0, ctypes.c_void_p(0))

    # Send the data over to the buffer (bytes)
    vs = vertices.tobytes()
    GL.glBufferData(GL.GL_ARRAY_BUFFER, len(vs), vs, GL.GL_STATIC_DRAW)

    # Generate buffers to hold our vertices
    color_buffer = GL.glGenBuffers(1)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, color_buffer)

    color = GL.glGetAttribLocation(shader, bytestr('color'))
    GL.glEnableVertexAttribArray(color)
    GL.glVertexAttribPointer(color, 3, GL.GL_FLOAT,
                             False, 0, ctypes.c_void_p(0))
    cs = colors.tobytes()
    GL.glBufferData(GL.GL_ARRAY_BUFFER, len(cs), cs, GL.GL_STATIC_DRAW)

    # Unbind the VAO first (Important)
    GL.glBindVertexArray(0)

    # Unbind other stuff
    GL.glDisableVertexAttribArray(position)
    GL.glDisableVertexAttribArray(color)
    GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    return vertex_array_object


def rot(a, b, c):
    s = np.sin(a)
    c = np.cos(a)
    am = np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)), np.float32)
    s = np.sin(b)
    c = np.cos(b)
    bm = np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)), np.float32)
    s = np.sin(c)
    c = np.cos(c)
    cm = np.array(((1, 0, 0), (0, c, s), (0, -s, c)), np.float32)
    return np.dot(np.dot(am, bm), cm)


def mobius_from_mouse(pos):

    shifted = np.array([pos.x - 960, pos.y - 540]).astype(np.float32)
    shifted /= 540
    snorm = np.linalg.norm(shifted)
    if snorm > 0.95:
        shifted *= 0.95/snorm

    a = np.array([1, 0], dtype=np.float32)
    b = np.array([shifted[0], -shifted[1]]).astype(np.float32)
    c = shifted
    d = np.array([1, 0], dtype=np.float32)

    return a, b, c, d


class ShaderFrame(pyopengltk.OpenGLFrame):

    def initgl(self):
        # GLUT.glutInit(sys.argv)
        GL.glClearColor(0, 0, 0, 1.0)
        # GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)
        if not hasattr(self, "shader"):
            self.shader = OpenGL.GL.shaders.compileProgram(
                compileShader(vertex_shader, GL.GL_VERTEX_SHADER),
                compileShader(fragment_shader, GL.GL_FRAGMENT_SHADER)
            )
            self.vertex_array_object = create_object(self.shader)
            self.proj = GL.glGetUniformLocation(self.shader, bytestr('proj'))

            self.a = GL.glGetUniformLocation(self.shader, bytestr('a'))
            self.b = GL.glGetUniformLocation(self.shader, bytestr('b'))
            self.c = GL.glGetUniformLocation(self.shader, bytestr('c'))
            self.d = GL.glGetUniformLocation(self.shader, bytestr('d'))

        self.nframes = 0
        self.start = time.time()

    def redraw(self):

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self.shader)
        t = time.time()-self.start

        p = np.stack([np.array([1, 0, 0]), np.array(
            [0, 1, 0]), np.array([0, 0, 1])])
        GL.glUniformMatrix3fv(self.proj, 1, GL.GL_FALSE, p)

        a, b, c, d = mobius_from_mouse(pyautogui.position())
        #a, c = np.array([1, 0]), np.array([1, 0])
        #b, d = np.zeros_like(a), np.zeros_like(c)
        GL.glUniform2f(self.a, a[0], a[1])
        GL.glUniform2f(self.b, b[0], b[1])
        GL.glUniform2f(self.c, c[0], c[1])
        GL.glUniform2f(self.d, d[0], d[1])

        GL.glBindVertexArray(self.vertex_array_object)
        GL.glDrawArrays(GL.GL_POINTS, 0, NPTS)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        GL.glRasterPos2f(0, 0)

        if self.nframes > 1:
            t = time.time()-self.start
            fps = "fps: %5.2f frames: %d" % (self.nframes / t, self.nframes)
            # for c in fps:
            #     GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_HELVETICA_18, ord(c));
        self.nframes += 1


def main():
    root = tk.Tk()
    app = ShaderFrame(root, width=1920, height=1080)
    app.pack(fill=tk.BOTH, expand=tk.YES)
    app.after(100, app.printContext)
    app.animate = 1000 // 60
    app.animate = 1
    app.mainloop()


if __name__ == '__main__':
    main()
