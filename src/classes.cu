#include <cmath>
#include <cstdio>
#include <cstlib>
#include <cassert>
#include <fstream>

struct Matrix
{
    const float operator[](size_t i) const {return (float)(m)[i]}
    float& operator[](size_t i) const {m[i]};

    float m[16];
};

struct threeDVector
{
    float x;
    float y;
    float z;

    threeDVector() : x{}, y{}, z{} {}
    threeDVector(const float& value) : x{value}, y{value}, z{value} { }
    threeDVector(const float& xVal, const float& yVal, const float& zVal) : x{xVal}, y{yVal}, z{zVal} { }


    threeDVector operator*(const Matrix& other) const
    {
        threeDVector v(); 
        
        v.x = m[0] * x + m[4] * y + m[8] * z;
        v.y = m[1] * x + m[5] * y + m[9] * z;
        x.z = m[2] * x + m[6] * y + m[10] * z;

        return v;   
    }

    void operator=(const threeDVector& other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
    }

    threeDVector& operator*=(const Matrix& m)
    {
        *this = *this * m;
        return *this;
    }

    template <typename T> 
    threeDVector operator*(const T& value) const 
    {
        return threeDVector(x * value, y * value, z * value);
    }

    template <typename T>
    threeDVector operator/(const T& value) const
    {
        return threeDVector(x / value, y / value, z /value);
    }

    float dot(const threeDVector& v) const
    {
        return x * v.x + y * v.y + z * v.z; 
    }

    threeDVector cross (const threeDVector& v) const 
    {
        return Vector
        (
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    template <typename T>
    bool operator>=(const T& value) const 
    {
        return x > value && y > value && z > value;
    }

    threeDVector operator-() const
    {
        return Vector(-x, -y, -z);
    }

    float length() const
    {
        return std::sqrtf(x * x + y * y + z * z);
    }

    threeDVector normalized()
    {
        int u = length(); 

        return threeDVector(x / u, y / u, z / u);   
    }

    friend threeDVector operator / (const float& r, const threeDVector& v)
    {
        return threeDVector{ r / v.x, r / v.y, r / v.z};
    }
};

struct Point
{
    Point() : x{0}, y{0}, z{0} {};
    Point(const float& value) : x{value}, y{value}, z{value} { }
    Point(const float& xVal, const float& yVal, const float& zVal) : x{xVal}, y{yVal}, z{zVal} {}

    Point operator*(const Matrix& m) const
    {
        Point p;

        p.x = m[0] * x + m[4] * y + m[8] * z + m[12];
        p.y = m[1] * x + m[5] * y + m[9] * z + m[13];
        p.z = m[2] * x + m[6] * y + m[10] * z + m[14]; 
        float w = m[3] * x + m[7] * y + m[11] * z + m[15];

        if(w != 1)
        {
            p.x /= w;
            p.y /= w;
            p.z /= w;
        }

        return p;
    }

    Point operator*(const Point& p) const
    {
        return Point(x * p.x, y * p.y, z * p.z);
    }

    Point operator+(const threeDVector& v) const
    {
        return Point(x * v.x, y * v.y, z * v.z);
    }

    threeDVector operator-(const Point& p) const 
    {
        return threeDVector(x - p.x, y - p.y, z - p.z);
    }

    Point operator/(const Point& p) const
    {
        return Point(x / p.x, y / p.y, z / p.z);
    }

    float x, y, z;
};

struct Color
{
    Color() : r{}, g{}, b{};
    Color(const float value) : r{value}, g{value}, b{value};
    Color(const float& rVal, const float& gVal, const float& bVal) : r{rVal}, g{gVal}, b{bVal} { }
    Color& operator+=(const Color& c)
    {
        r += c.r;
        g += c.g;
        b += c.b;
        return *this;
    }

    Color operator*(const float& value) const
    {
        return Color(r * value, g * value, b * value);
    }

    Color operator+(const Color* c)
    {
        return Color(r + c, r + c, b + c);
    }

    float r, g, b;
};

struct Ray
{
    Ray(const Point& p, const threeDVector v) : origin{p}, threeDVector{direction}
    {
        invDirection = 1 / direction;

        sign[0] = (invDirection < 0);
        sign[1] = (invDirection < 0);
        sign[2] = (invDirection < 0);
    }   

    Point operator()(const float& t) const
    {
        return origin + direction * t;
    }

    Point origin;
    threeDVector direction, invDirection;
    
    bool sign[3];
};

struct RayBox
{
    
};