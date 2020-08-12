#ifndef SHADER_H
#define SHADER_H

#include <string>
#include "../Utils/EigenDef.h"

class Shader
{
public:
	Shader(const std::string& vertexPath, const std::string& fragmentPath);
	~Shader() {};

	unsigned int getID() const { return m_ID; }

	void use();
	
	void setBool(const std::string& name, bool value) const;
	void setInt(const std::string& name, int value) const;
	void setFloat(const std::string& name, float value) const;
	void setVec4f(const std::string& name, float x, float y, float z, float w) const;
	void setMat4f(const std::string& name, const Eigen::Mat4f& value) const;

private:
	void checkCompileErrors(unsigned int shader, std::string type);

	unsigned int m_ID;
};


#endif // !SHADER_H
