#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec4 inTangent;

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

layout (location = 0) out vec2 fragTexCoord;
layout (location = 1) out vec3 fragNormal;
layout (location = 2) out vec3 fragPosition;
layout (location = 3) out mat3 TBN;
layout (location = 8) out vec4 fragPositionLightSpace;

void main() {
	//calculate TBN matrix
	vec3 T = normalize(inTangent.xyz * inTangent.w);
	vec3 N = normalize(inNormal);
	vec3 B = normalize(cross(N, T));
	TBN = mat3(T, B, N);

    gl_Position = uScene.projcam * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    fragNormal = inNormal;
    fragPosition = inPosition;
	fragPositionLightSpace = uScene.lightSpaceMatrx * vec4(inPosition, 1.0f);
}
