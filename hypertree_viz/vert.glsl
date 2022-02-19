#version 130
in vec3 position;
in vec3 color;

uniform mat3 proj;

uniform vec2 a;
uniform vec2 b;
uniform vec2 c;
uniform vec2 d;

varying vec3 vert_color;

vec2 cinv(vec2 z) {
    return vec2(z.x, -z.y)/(z.x*z.x + z.y*z.y);
}

vec2 cmul(vec2 z1, vec2 z2) {
    return vec2(z1.x*z2.x - z1.y*z2.y, z1.x*z2.y + z1.y*z2.x);
}

vec2 mobius(vec2 z) {
    z.x *= 1.8;
    z = cmul(cmul(a, z) + b, cinv(cmul(c, z) + d));
    //z = cmul(a, z) + b;
    z.x *= 0.555;
    return z;
}

void main()
{
   vec3 transformed = vec3(mobius(position.xy), position.z);
   //vec3 transformed = position;
   gl_Position = vec4(proj*transformed, 1);
   vert_color = color;
}