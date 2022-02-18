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


vertex_shader = """#version 130
in vec3 position;
in vec3 color;
varying vec3 vert_color;
uniform mat3 proj;
//*
uniform mat2 rmobius;
uniform mat2 imobius;

vec2 cinv(vec2 z) {
    return vec2(z.x, -z.y)/dot(z, z);
}

vec2 cmul(vec2 a, vec2 z) {
    return vec2(a.x*z.x - a.y*z.y, a.x*z.y + a.y*z.x);
}
//*/
void main()
{
   //*
   vec2 rtransformed = rmobius*vec2(position.x, 1) - imobius*vec2(position.y, 0);
   vec2 itransformed = rmobius*vec2(position.y, 0) + imobius*vec2(position.x, 1);
   vec2 inv2 = cinv(vec2(rtransformed.y, itransformed.y));
   vec2 transformed = cmul(inv2, vec2(rtransformed.x, itransformed.y));
   //*/
   gl_Position = vec4(transformed, 1, 1);
   //gl_Position = vec4(proj*position, 1);
   vert_color = color;
}
"""

fragment_shader = """#version 130
varying vec3 vert_color;
void main()
{
   gl_FragColor = vec4(vert_color, 1);
}
"""
NPTS = 100000

#vertices = (np.random.random(NPTS * 3).astype(np.float32)-.5) * 1.5
#vertices.shape = NPTS, 3

print('Loading vertices . . .')
with open('/home/medusa/Projects/hyperbolic-embedder/billeh-coordinates.txt', 'r') as f:
    lines = f.readlines()
    NPTS = len(lines) - 2
    vertices = np.zeros((NPTS, 3), dtype=np.float32)
    perm = np.zeros(NPTS, dtype=np.int32)
    for i, line in tqdm(enumerate(lines[2:])):
        coordstrings = line.split()
        perm[i] = int(coordstrings[0])
        r, phi = float(coordstrings[1]) / \
            13.91, np.pi*float(coordstrings[2])/180
        vertices[i] = np.array([(5/9)*r*np.cos(phi), r*np.sin(phi), 1])

colors = np.zeros((NPTS, 3), dtype=np.float32)
print('Finding neuron types . . .')
with open('/home/medusa/Projects/billeh_tinkering/neuron_types.pkl', 'rb') as f:
    neuron_types = pkl.load(f)['neuron_type_names']
with open('/home/medusa/Projects/billeh_tinkering/network_dat_original.pkl', 'rb') as f:
    nodes = pkl.load(f)['nodes']
    for typename, node in tqdm(zip(neuron_types, nodes)):
        if typename.startswith('e') and '1' in typename:
            for id in node['ids']:
                ix = np.where(perm == id)[0]
                colors[ix] = np.array([1, 0, 0])
        elif typename.startswith('e') and '23' in typename:
            for id in node['ids']:
                ix = np.where(perm == id)[0]
                colors[ix] = np.array([1, 0.25, 0])
        elif typename.startswith('e') and '4' in typename:
            for id in node['ids']:
                ix = np.where(perm == id)[0]
                colors[ix] = np.array([1, 0.5, 0])
        elif typename.startswith('e') and '5' in typename:
            for id in node['ids']:
                ix = np.where(perm == id)[0]
                colors[ix] = np.array([1, 0.75, 0])
        elif typename.startswith('e') and '6' in typename:
            for id in node['ids']:
                ix = np.where(perm == id)[0]
                colors[ix] = np.array([1, 1, 0])

        elif typename.startswith('i') and '1' in typename:
            for id in node['ids']:
                ix = np.where(perm == id)[0]
                colors[ix] = np.array([0, 0, 1])
        elif typename.startswith('i') and '23' in typename:
            for id in node['ids']:
                ix = np.where(perm == id)[0]
                colors[ix] = np.array([0, 0.25, 1])
        elif typename.startswith('i') and '4' in typename:
            for id in node['ids']:
                ix = np.where(perm == id)[0]
                colors[ix] = np.array([0, 0.5, 1])
        elif typename.startswith('i') and '5' in typename:
            for id in node['ids']:
                ix = np.where(perm == id)[0]
                colors[ix] = np.array([0, 0.75, 1])
        elif typename.startswith('i') and '6' in typename:
            for id in node['ids']:
                ix = np.where(perm == id)[0]
                colors[ix] = np.array([0, 1, 1])


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
    GL.glVertexAttribPointer(color, 3, GL.GL_FLOAT, False, 0, ctypes.c_void_p(0))
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

    shifted = np.array([pos.x - 960, pos.y - 540]).astype(np.float64)
    snorm = np.linalg.norm(shifted)
    if snorm > 540:
        shifted /= snorm
    else:
        shifted /= 540
    cpos = shifted[0] + shifted[1]*1j

    r1 = np.array([1, -cpos], dtype=np.complex64)
    r2 = np.array([cpos, 1], dtype=np.complex64)
    transformation = np.stack([r1, r2], axis=0)

    return transformation.real.astype(np.float32), transformation.imag.astype(np.float32)


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
            self.rmobius = GL.glGetUniformLocation(self.shader, bytestr('rmobius'))
            self.imobius = GL.glGetUniformLocation(self.shader, bytestr('imobius'))
        self.nframes = 0
        self.start = time.time()

    def redraw(self):

        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        GL.glUseProgram(self.shader)
        t = time.time()-self.start

        p = np.stack([np.array([1, 0, 0]), np.array(
            [0, 1, 0]), np.array([0, 0, 1])])
        GL.glUniformMatrix3fv(self.proj, 1, GL.GL_FALSE, p)

        new_rmobius, new_imobius = mobius_from_mouse(pyautogui.position())
        GL.glUniformMatrix2fv(self.rmobius, 1, GL.GL_FALSE, new_rmobius)
        GL.glUniformMatrix2fv(self.imobius, 1, GL.GL_FALSE, new_imobius)

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
