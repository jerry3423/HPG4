#version 450

layout (location = 0) in vec3 inPosition;

struct Light
{
	vec4 lightPosition;
	vec4 lightColor;
};

layout(set = 0, binding = 0) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projcam;
	vec4 cameraPosition;
	Light pointLight[2];
	mat4 lightSpaceMatrx;
} uScene;


void main()
{
    vec4 lightpos  = uScene.lightSpaceMatrx  * vec4(inPosition, 1.0);
	gl_Position = vec4(lightpos.x, lightpos.y, lightpos.z + 0.1f, lightpos.w);
}
