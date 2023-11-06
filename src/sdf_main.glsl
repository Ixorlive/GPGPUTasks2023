struct LEG
{
    vec3 point;
    vec3 knee;
    vec3 ankle;
    vec3 fix;
};

const int NUM_LEGS = 8;
LEG legs[NUM_LEGS];

const vec3 sky_color = vec3(135.0 / 255.0, 206.0 / 255.0, 250.0 / 255.0);
const vec3 grass_color = vec3(0.25, 0.5, 0.2);
const vec3 spider_color = vec3(0.4, 0.26, 0.13);

float smin( float a, float b, float k /*== 32*/)
{
	float h = clamp(0.5 + 0.5*(b-a)/k, 0.0, 1.0 );
	return mix( b, a, h ) - k*h*(1.-h);
}

// косинус который пропускает некоторые периоды, удобно чтобы махать ручкой не все время
float lazycos(float angle)
{
    int nsleep = 10;
    
    int iperiod = int(angle / 6.28318530718) % nsleep;
    if (iperiod < 3) {
        return cos(angle);
    }
    
    return 1.0;
}

// sphere with center in (0, 0, 0)
float sdSphere(vec3 p, float r)
{
    return length(p) - r;
}


float upperLeg( vec3 p, vec3 a, vec3 b, float r )
{
	vec3 pa = p-a, ba = b-a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return length( pa - ba*h ) - r*(sin(h*2.14+.4));
}

float lowerLeg(vec3 p,  vec3 a, vec3 b, float r1, float r2)
{
	vec3 pa = p - a;
	vec3 ba = b - a;
	float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
	return length( pa - ba*h ) - r1 + r2*h;
}


// XZ plane
float sdPlane(vec3 p)
{
    return p.y;
}


float sdCapsule(vec3 p, float s )
{
    return length(p* vec3(1., 1.0, .8)) - s;
}

// возможно, для конструирования тела пригодятся какие-то примитивы из набора https://iquilezles.org/articles/distfunctions/
// способ сделать гладкий переход между примитивами: https://iquilezles.org/articles/smin/
vec4 sdBody(vec3 p)
{   
    float ass = sdCapsule(p - vec3(0.0, 0.45, -1.0), .6);
    float head = sdSphere(p - vec3(0.0, 0.35, 0.5), .26);
    float body = smin(ass, head, .36);
    float d = body;
    
    for (int i = 0; i < NUM_LEGS; i++)
    {
        d = min(d, upperLeg(p, legs[i].fix, legs[i].knee, .08)); 
        d = min(d, upperLeg(p, legs[i].knee, legs[i].ankle, .09)); 
        d = min(d, lowerLeg(p, legs[i].ankle, legs[i].point, .1, .08)); 
    }
    
    p.x = abs(p.x);
    d = min(d, lowerLeg(p, vec3(0.1, 0.25, 0.7), vec3(0.07, -.4, 0.5), .02,.05));
    
    return vec4(d, spider_color);
}


vec4 sdEye(vec3 p)
{
    float d = 1e10;
    
    vec3 col = vec3(.0, .0, .0);
    
    float offset_z = 0.003 * (sin(iTime*2.0 - 3.141/2.0) + 1.0);
    float offset_x = 0.018 * sin(iTime);
    
    float pupilLeft = sdSphere(p - vec3(-0.1 - offset_x, 0.39, 0.732 - offset_z), .05);
    float pupilRight = sdSphere(p - vec3(0.1 - offset_x, 0.39, 0.732 - offset_z), .05);

    d = min(d, pupilLeft);
    d = min(d, pupilRight);
    
    p.x = abs(p.x);
    d = min(d, sdSphere(p - vec3(0.1, 0.39, 0.7 ), .08));
    
    if (d >= pupilLeft || d >= pupilRight) {
        col = vec3(1.0, 1.0, 1.0);
    }
    return vec4(d, col);
}

vec4 sdMonster(vec3 p)
{
    // при рисовании сложного объекта из нескольких SDF, удобно на верхнем уровне 
    // модифицировать p, чтобы двигать объект как целое
    p -= vec3(0.0, 0.08, 0.0);
    
    vec4 res = sdBody(p); 
    
    vec4 eye = sdEye(p);
    if (eye.x < res.x) {
        res = eye;
    }
    
    return res;
}


vec4 sdTotal(vec3 p)
{
    vec4 res = sdMonster(p);
    
    
    float dist = sdPlane(p);
    if (dist < res.x) {
        res = vec4(dist, grass_color);
    }
    
    return res;
}


// see https://iquilezles.org/articles/normalsSDF/
vec3 calcNormal( in vec3 p ) // for function f(p)
{
    const float eps = 0.0001; // or some other value
    const vec2 h = vec2(eps,0);
    return normalize( vec3(sdTotal(p+h.xyy).x - sdTotal(p-h.xyy).x,
                           sdTotal(p+h.yxy).x - sdTotal(p-h.yxy).x,
                           sdTotal(p+h.yyx).x - sdTotal(p-h.yyx).x ) );
}


vec4 raycast(vec3 ray_origin, vec3 ray_direction)
{
    
    float EPS = 1e-3;
    
    
    // p = ray_origin + t * ray_direction;
    
    float t = 0.0;
    
    for (int iter = 0; iter < 200; ++iter) {
        vec4 res = sdTotal(ray_origin + t*ray_direction);
        t += res.x;
        if (res.x < EPS) {
            return vec4(t, res.yzw);
        }
    }

    return vec4(1e10, sky_color);
}


float shading(vec3 p, vec3 light_source, vec3 normal)
{
    
    vec3 light_dir = normalize(light_source - p);
    
    float shading = dot(light_dir, normal);
    
    return clamp(shading, 0.5, 1.0);

}

// phong model, see https://en.wikibooks.org/wiki/GLSL_Programming/GLUT/Specular_Highlights
float specular(vec3 p, vec3 light_source, vec3 N, vec3 camera_center, float shinyness)
{
    vec3 L = normalize(p - light_source);
    vec3 R = reflect(L, N);

    vec3 V = normalize(camera_center - p);
    
    return pow(max(dot(R, V), 0.0), shinyness);
}


float castShadow(vec3 p, vec3 light_source)
{
    
    vec3 light_dir = p - light_source;
    
    float target_dist = length(light_dir);
    
    
    if (raycast(light_source, normalize(light_dir)).x + 0.001 < target_dist) {
        return 0.5;
    }
    
    return 1.0;
}


void initializeLegs()
{
    for (int i = 0; i < NUM_LEGS; i++)
    {
        float sign = (i % 2 == 0) ? 1.0 : -1.0;
        float frontBackMultiplier = (i < 4) ? 1.0 : -1.0;
        float lateralOffset = sign * (0.17 + 0.05 * float(i / 2));
        float frontBackOffset = 0.3 + frontBackMultiplier * (0.3 + 0.05 * float(i / 2));
       
        legs[i].fix = vec3(lateralOffset, 0.25, frontBackOffset);
        legs[i].point = vec3(lateralOffset + sign * 1.0, 0.0, frontBackOffset);
        legs[i].knee = vec3(lateralOffset * 2.0, 0.8, frontBackOffset);
        legs[i].ankle = vec3(lateralOffset * 3.5, 0.8, frontBackOffset);
    }
}

void animate() {
    for (int i = 0; i < NUM_LEGS; i++)
    {
        float moveFactor1 = sin(2.0*iTime + float(i) * 0.5) * 0.07;
        // lazycos for fun
        float moveFactor2 = lazycos(2.0*iTime + float(i) * 0.3) * 0.07;
        float moveFactor3 = sin(2.0*iTime + float(i) * 0.7) * 0.07;

        legs[i].knee.y += moveFactor1;
        legs[i].ankle.y += moveFactor2;
        legs[i].point.y += moveFactor3;

        legs[i].knee = clamp(legs[i].knee, vec3(-2.0, 0.0, -2.0), vec3(2.0, 2.0, 2.0));
        legs[i].ankle = clamp(legs[i].ankle, vec3(-2.0, 0.0, -2.0), vec3(2.0, 2.0, 2.0));
        legs[i].point = clamp(legs[i].point, vec3(-2.0, 0.0, -2.0), vec3(2.0, 2.0, 2.0));
    }
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    initializeLegs();
    animate();

    vec2 uv = fragCoord/iResolution.y;
    
    vec2 wh = vec2(iResolution.x / iResolution.y, 1.0);
    

    vec3 ray_origin = vec3(0.35, 0.8, 2.9);
    vec3 ray_direction = normalize(vec3(uv - 0.5*wh, -1.0));
    
    vec4 res = raycast(ray_origin, ray_direction);
   
    vec3 col = res.yzw;
    
    
    vec3 surface_point = ray_origin + res.x*ray_direction;
    vec3 normal = calcNormal(surface_point);
    
    vec3 light_source = vec3(1.0 + 2.5*sin(iTime), 10.0, 10.0);
    
    float shad = shading(surface_point, light_source, normal);
    shad = min(shad, castShadow(surface_point, light_source));
    col *= shad;
    
    float spec = specular(surface_point, light_source, normal, ray_origin, 30.0);
    col += vec3(1.0, 1.0, 1.0) * spec;
    
    
    
    // Output to screen
    fragColor = vec4(col, 1.0);
}
