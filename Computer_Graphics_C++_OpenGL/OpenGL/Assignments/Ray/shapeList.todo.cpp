#include <Util/exceptions.h>
#include <Util/threads.h>
#include "shapeList.h"
#include "triangle.h"

using namespace Ray;
using namespace Util;

////////////////
// Difference //
////////////////
void Difference::updateBoundingBox( void )
{
	///////////////////////////////
	// Set the _bBox object here //
	///////////////////////////////
	_shape0->updateBoundingBox();
	_bBox = _shape0->boundingBox();
}

bool Difference::processFirstIntersection( const Ray3D &ray , const BoundingBox1D &range , const RayIntersectionFilter &rFilter , const RayIntersectionKernel &rKernel , ShapeProcessingInfo spInfo , unsigned int tIdx ) const
{
	//////////////////////////////////////////////////////////////////
	// Compute the intersection of the difference with the ray here //
	//////////////////////////////////////////////////////////////////
	WARN_ONCE( "method undefined" );
	return false;
}

int Difference::processAllIntersections( const Ray3D &ray , const BoundingBox1D &range , const RayIntersectionFilter &rFilter , const RayIntersectionKernel &rKernel , ShapeProcessingInfo spInfo , unsigned int tIdx ) const
{
	//////////////////////////////////////////////////////////////////
	// Compute the intersection of the difference with the ray here //
	//////////////////////////////////////////////////////////////////
	WARN_ONCE( "method undefined" );
	return 0;
}
bool Difference::isInside( Util::Point3D p ) const
{
	//////////////////////////////////////////////////////////
	// Determine if the point is inside the difference here //
	//////////////////////////////////////////////////////////
	WARN_ONCE( "method undefined" );
	return false;
}

///////////////
// ShapeList //
///////////////
bool ShapeList::processFirstIntersection( const Ray3D &ray , const BoundingBox1D &range , const RayIntersectionFilter &rFilter , const RayIntersectionKernel &rKernel , ShapeProcessingInfo spInfo , unsigned int tIdx ) const
{
	//////////////////////////////////////////////////////////////////
	// Compute the intersection of the shape list with the ray here //
	//////////////////////////////////////////////////////////////////

	RayShapeIntersectionInfo rsiInfo;
	bool hit = false;

	RayIntersectionKernel k = [&rsiInfo, &spInfo, &hit]( const ShapeProcessingInfo &_spInfo , const RayShapeIntersectionInfo &_iInfo ) {

		if (_iInfo.t < rsiInfo.t) {
			rsiInfo = _iInfo;
			spInfo = _spInfo;
			hit = true;
			return true;
		}
		return false;
	};


	for( unsigned int i = 0; i < shapes.size(); i++) {
		shapes[i]->processFirstIntersection(ray, range, rFilter, k, spInfo, tIdx);
	}
	if (hit) {
		rKernel(spInfo, rsiInfo);
	}

	return hit;
}

int ShapeList::processAllIntersections( const Ray3D &ray , const BoundingBox1D &range , const RayIntersectionFilter &rFilter , const RayIntersectionKernel &rKernel , ShapeProcessingInfo spInfo , unsigned int tIdx ) const
{
	int count = 0;
	double transparency = 1;
	// RayIntersectionKernel k = [&] (const ShapeProcessingInfo& spInfo, double &transparency) {
	// 	transparency *= spInfo.material->transparent;
	// 	return true;
	// };
	for( unsigned int i=0 ; i<shapes.size() ; i++ ) count += shapes[i]->processAllIntersections( ray , range , rFilter , rKernel , spInfo , tIdx );
	return count;
}
bool ShapeList::isInside( Point3D p ) const
{
	//////////////////////////////////////////////////////////
	// Determine if the point is inside the shape list here //
	//////////////////////////////////////////////////////////
	WARN_ONCE( "method undefined" );
	return false;
}

void ShapeList::init( const LocalSceneData &data )
{
	// Initialize the children
	for( int i=0 ; i<shapes.size() ; i++ ) shapes[i]->init( data );
	_primitiveNum = 0;
	for( int i=0 ; i<shapes.size() ; i++ ) _primitiveNum += shapes[i]->primitiveNum();

	///////////////////////////////////
	// Do any additional set-up here //
	///////////////////////////////////
	WARN_ONCE( "method undefined" );
}

void ShapeList::updateBoundingBox( void )
{
	///////////////////////////////
	// Set the _bBox object here //
	///////////////////////////////
	Point3D *pList = new Point3D[shapes.size()*2];
	for( int i=0 ; i<shapes.size() ; i++ )
	{
		shapes[i]->updateBoundingBox();
		_bBox = shapes[i]->boundingBox();
		pList[2*i  ] = _bBox[0];
		pList[2*i+1] = _bBox[1];
	}
	_bBox = BoundingBox3D( pList , (int)shapes.size()*2 );
	delete[] pList;
}

void ShapeList::initOpenGL( void )
{
	// Initialize the children
	for( int i=0 ; i<shapes.size() ; i++ ) shapes[i]->initOpenGL();

	///////////////////////////
	// Do OpenGL set-up here //
	///////////////////////////
	WARN_ONCE( "remainder of method undefined" );
}

void ShapeList::drawOpenGL( GLSLProgram * glslProgram ) const
{
	//////////////////////////////
	// Do OpenGL rendering here //
	//////////////////////////////
	for (int i = 0; i < shapes.size(); i++) {
		shapes[i]->drawOpenGL(glslProgram);
	}

	// Sanity check to make sure that OpenGL state is good
	ASSERT_OPEN_GL_STATE();	
}

/////////////////
// AffineShape //
/////////////////

bool AffineShape::isInside( Point3D p ) const
{
	///////////////////////////////////////////////////////////////////////
	// Determine if the point is inside the affinely deformed shape here //
	///////////////////////////////////////////////////////////////////////
	WARN_ONCE( "method undefined" );
	return _shape->isInside( p );
}

void AffineShape::updateBoundingBox( void )
{
	///////////////////////////////
	// Set the _bBox object here //
	///////////////////////////////
	_shape->updateBoundingBox();
	_bBox = getMatrix() * _shape->boundingBox();
}

void AffineShape::drawOpenGL( GLSLProgram * glslProgram ) const
{
	//////////////////////////////
	// Do OpenGL rendering here //
	//////////////////////////////

	// Get the transformation matrix from getMatrix()
    Util::Matrix4D transformationMatrix = getMatrix();

    // Convert the glm::mat4 to a GLfloat array
    GLfloat matrixArray[16];
	int index = 0;

	for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            matrixArray[i * 4 + j] = transformationMatrix(j, i);
        }
    }

    // Push the transformation matrix onto the OpenGL matrix stack
    glPushMatrix();
    glMultMatrixf(matrixArray);

    // Call drawOpenGL for the underlying shape
    _shape->drawOpenGL(glslProgram);

    // Pop the transformation matrix off the stack when done
    glPopMatrix();

    // Sanity check to make sure that OpenGL state is good
    ASSERT_OPEN_GL_STATE();
}

///////////////////////
// StaticAffineShape //
///////////////////////
void StaticAffineShape::init( const LocalSceneData &data )
{
	////////////////////////////////////////
	// Do/replace the initialization here //
	////////////////////////////////////////
	_inverseTransform = _localTransform.inverse();
	_normalTransform = Util::Matrix3D(_localTransform);
	_normalTransform = _normalTransform.inverse().transpose();

	_shape->init( data );
	_primitiveNum = _shape->primitiveNum();
}

//////////////////
// TriangleList //
//////////////////

void TriangleList::drawOpenGL( GLSLProgram * glslProgram ) const
{
	//////////////////////////////
	// Do OpenGL rendering here //
	//////////////////////////////

    // Loop through each shape in the list
    for (int i = 0; i < shapes.size(); i++) {
        shapes[i]->drawOpenGL(glslProgram);
    }

	// Sanity check to make sure that OpenGL state is good
    ASSERT_OPEN_GL_STATE();
}

void TriangleList::init( const LocalSceneData &data )
{
	// Set the vertex and material pointers
	_vertices = &data.vertices[0];
	_vNum = (unsigned int)data.vertices.size();
	if( _materialIndex>=data.materials.size() ) THROW( "shape specifies a material that is out of bounds: " , _materialIndex , " <= " , data.materials.size() );
	else if( _materialIndex<0 ) THROW( "negative material index: " , _materialIndex );
	else _material = &data.materials[ _materialIndex ];

	ShapeList::init( data );

	///////////////////////////////////
	// Do any additional set-up here //
	///////////////////////////////////
	WARN_ONCE( "method undefined" );
}

void TriangleList::initOpenGL( void )
{
	ShapeList::initOpenGL();

	// Initialize the vertex and element buffer objects
	std::vector< TriangleIndex > triangles;
	addTrianglesOpenGL( triangles );
	_tNum = (unsigned int)triangles.size();

	GLfloat* vertexData = new GLfloat[ _vNum * ( 3 + 3 + 2 ) ];
	GLfloat *v = vertexData + _vNum * 0;
	GLfloat* n = vertexData + _vNum * 3;
	GLfloat* t = vertexData + _vNum * 6;

	for( int i=0 ; i<(int)_vNum ; i++ )
	{
		for( int j=0 ; j<3 ; j++ ) v[3*i+j] = (GLfloat)_vertices[i].position[j];
		for( int j=0 ; j<3 ; j++ ) n[3*i+j] = (GLfloat)_vertices[i].normal[j];
		for( int j=0 ; j<2 ; j++ ) t[2*i+j] = (GLfloat)_vertices[i].texCoordinate[j];
	}
	glGenBuffers( 1 , &_vertexBufferID );
	glBindBuffer( GL_ARRAY_BUFFER , _vertexBufferID );
	glBufferData( GL_ARRAY_BUFFER , 8 * _vNum * sizeof( GLfloat ) , vertexData , GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER , 0 );
	delete[] vertexData;

	glGenBuffers( 1 , &_elementBufferID );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER , _elementBufferID );
	glBufferData( GL_ELEMENT_ARRAY_BUFFER , triangles.size() * sizeof(GLuint) * 3 , &triangles[0][0] , GL_STATIC_DRAW );
	glBindBuffer( GL_ELEMENT_ARRAY_BUFFER , 0 );

	///////////////////////////
	// Do OpenGL set-up here //
	///////////////////////////
	WARN_ONCE( "method undefined" );

	// Sanity check to make sure that OpenGL state is good
	ASSERT_OPEN_GL_STATE();	
}

///////////
// Union //
///////////
bool Union::processFirstIntersection( const Ray3D &ray , const BoundingBox1D &range , const RayIntersectionFilter &rFilter , const RayIntersectionKernel &rKernel , ShapeProcessingInfo spInfo , unsigned int tIdx ) const
{
	/////////////////////////////////////////////////////////////
	// Compute the intersection of the union with the ray here //
	/////////////////////////////////////////////////////////////
	WARN_ONCE( "method undefined" );
	return false;
}

int Union::processAllIntersections( const Ray3D &ray , const BoundingBox1D &range , const RayIntersectionFilter &rFilter , const RayIntersectionKernel &rKernel , ShapeProcessingInfo spInfo , unsigned int tIdx ) const
{
	/////////////////////////////////////////////////////////////
	// Compute the intersection of the union with the ray here //
	/////////////////////////////////////////////////////////////
	WARN_ONCE( "method undefined" );
	return 0;
}
void Union::init( const LocalSceneData &data )
{
	_shapeList.init( data );
	_primitiveNum = _shapeList.primitiveNum();

	///////////////////////////////////
	// Do any additional set-up here //
	///////////////////////////////////
	WARN_ONCE( "method undefined" );
}

void Union::updateBoundingBox( void )
{
	///////////////////////////////
	// Set the _bBox object here //
	///////////////////////////////
	_shapeList.updateBoundingBox();
	_bBox = _shapeList.shapes[0]->boundingBox();
	for( int i=1 ; i<_shapeList.shapes.size() ; i++ ) _bBox += _shapeList.shapes[i]->boundingBox();
}

bool Union::isInside( Point3D p ) const
{
	/////////////////////////////////////////////////////
	// Determine if the point is inside the union here //
	/////////////////////////////////////////////////////
	WARN_ONCE( "method undefined" );
	return false;
}

//////////////////
// Intersection //
//////////////////
bool Intersection::processFirstIntersection( const Ray3D &ray , const BoundingBox1D &range , const RayIntersectionFilter &rFilter , const RayIntersectionKernel &rKernel , ShapeProcessingInfo spInfo , unsigned int tIdx ) const
{
	/////////////////////////////////////////////////////////////////////////////////////
	// Compute the intersection of the difference with the intersection of shapes here //
	/////////////////////////////////////////////////////////////////////////////////////
	WARN_ONCE( "method undefined" );
	return false;
}

int Intersection::processAllIntersections( const Ray3D &ray , const BoundingBox1D &range , const RayIntersectionFilter &rFilter , const RayIntersectionKernel &rKernel , ShapeProcessingInfo spInfo , unsigned int tIdx ) const
{
	/////////////////////////////////////////////////////////////////////////////////////
	// Compute the intersection of the difference with the intersection of shapes here //
	/////////////////////////////////////////////////////////////////////////////////////
	WARN_ONCE( "method undefined" );
	return 0;
}
void Intersection::init( const LocalSceneData &data )
{
	_shapeList.init( data );
	_primitiveNum = _shapeList.primitiveNum();

	///////////////////////////////////
	// Do any additional set-up here //
	///////////////////////////////////
	WARN_ONCE( "method undefined" );
}

void Intersection::updateBoundingBox( void )
{
	///////////////////////////////
	// Set the _bBox object here //
	///////////////////////////////
	_shapeList.updateBoundingBox();
	_bBox = _shapeList.shapes[0]->boundingBox();
	for( int i=1 ; i<_shapeList.shapes.size() ; i++ ) _bBox ^= _shapeList.shapes[i]->boundingBox();
}

bool Intersection::isInside( Point3D p ) const
{
	///////////////////////////////////////////////////////////////////////
	// Determine if the point is inside the instersection of shapes here //
	///////////////////////////////////////////////////////////////////////
	WARN_ONCE( "method undefined" );
	return false;
}