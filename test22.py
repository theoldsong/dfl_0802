import os
#os.environ["DFL_PLAIDML_BUILD"] = "1"
import pickle
import math
import sys
import argparse
from core import pathex
from core import osex
from facelib import LandmarksProcessor
from facelib import FaceType
from pathlib import Path
import numpy as np
from numpy import linalg as npla
import cv2
import time
import multiprocessing
import threading
import traceback
from tqdm import tqdm
from DFLIMG import *
from core.cv2ex import *
import shutil
from core import imagelib
from core.interact import interact as io

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from core.qtex import *
from OpenGL import GL as gl
import ctypes
from enum import IntEnum
import uuid


class NodeType(IntEnum):
    HALF = 0
    
    @staticmethod
    def fromString (s):
        r = from_string_dict.get (s.lower())
        if r is None:
            raise Exception ('FaceType.fromString value error')
        return r

    @staticmethod
    def toString (face_type):
        return to_string_dict[face_type]

to_string_dict = { FaceType.HALF : 'half_face',
                 }

from_string_dict = { to_string_dict[x] : x for x in to_string_dict.keys() }  



"""

b = sdBox    ( p, params )
s = sdSphere ( p, params )

(dist, mat)

"""


vertex_code = '''
attribute vec2 position;
void main()
{
  gl_Position = vec4(position, 0.0, 1.0);
}
'''


fragment_code = \
'''
#define FLT_MAX 3.402823466e+38
#define FLT_MIN 1.175494351e-38

const float PI = 3.1415926535897932384626433832795;


struct SD
{
    float dist;
    float material;
};

#define MATERIAL_NONE -3.0
#define MATERIAL_DIST_MAX -2.0
#define MATERIAL_ITER_MAX -1.0
#define MATERIAL_FLOOR 1.0
#define MATERIAL_SPHERE 2.0

uniform vec2 iResolution;

float sdBox( vec3 pos, vec3 size )
{
    vec3 d = abs(pos) - size;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

SD part_1 ( vec3 pos )
{
    // 
    //
    //
    //
    
    float dist = sdBox( pos, vec3(1.0,1.0,1.0) );
    
    return  SD (dist, MATERIAL_SPHERE );
}

SD scene_map( vec3 pos )
{
    SD result = part_1(pos)
    
	return result;
}
    
    
vec3 scene_render ( vec3 ro, vec3 rd )
{	
    float depth_min = 0.01;
    float depth_max = 150.0;

    float depth = depth_min;
    
    vec3 hit_pos;
    
    float mat_id;

    mat_id = MATERIAL_ITER_MAX;
    
    for( int i=0; i<50; i++ )
    {
	    float hit_precis = 0.0002*depth;
        hit_pos = ro+rd*depth;
        
        SD sd = scene_map( hit_pos );

        if( sd.dist < hit_precis)
        {
            mat_id = sd.material;
            break;
        }
        
        if( depth > depth_max )
        {
            mat_id = MATERIAL_DIST_MAX;
            break;
        }
        
        depth += sd.dist;
        
    } 
    
    if (mat_id == MATERIAL_FLOOR)
    {
        return vec3(1.0,1.0,1.0);
    } else
    if (mat_id == MATERIAL_SPHERE)
    {
        return vec3(1.0,1.0,0.0);
    }
    if (mat_id == MATERIAL_ITER_MAX)
    {        
    	return vec3(1.0,0.0,1.0);
    }
    
   	return vec3(0.0,0.0,0.0);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord.xy / iResolution.xy) * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y;
        
    vec3 vd = normalize(vec3(uv, 1.5));
    vec3 color = scene_render ( vec3(0.0,0,-5), vd );
    
    fragColor = vec4(color, 1.0);
    
    //fragColor = vec4( fragCoord.x * (1.0/iResolution.x), 0.0, 0.0, 1.0);
}

void main() { mainImage ( gl_FragColor, gl_FragCoord ); }
'''

class Serializable():
    #override
    def serialize():
        return {}
    
class GUIDable(Serializable):
    def __init__(self, guid=None, **kwargs):
        if guid is None:
            guid = uuid.uuid4().hex
        self.guid = guid

    def get_guid(self):
        return self.guid
        
    #override
    def serialize(self):
        d = super().serialize()
        d.update ( {'guid':self.guid} )
        return d
        
        
class NODE_CONN_TYPE(IntEnum):
    NONE = 0
    
    FLOAT = 1
    
    VEC2 = 10
    VEC3 = 11
    VEC4 = 12
    
    @staticmethod
    def fromString (s):
        r = from_string_dict.get (s.lower())
        if r is None:
            raise Exception ('fromString value error')
        return r

    #def serialize ():
    #    return node_connection_type_dict[]
        
  
class NodeConnSlot():
    __slots__ = ['name', 'conn_type']
    def __init__(self, name, conn_type):
        self.name = name
        self.conn_type = conn_type
    
class NodeConnection(GUIDable):
    def __init__(self, node1, node1_out_slot_name, node2, node2_in_slot_name, **kwargs):
        super().__init__(**kwargs)
        self.node1 = node1
        self.node1_out_slot_name = node1_out_slot_name
        self.node2 = node2
        self.node2_in_slot_name = node2_in_slot_name

class Node(GUIDable):
    def __init__(self, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.conns = []

    #override
    def get_in_conn_slots(self):
        # return list of NodeConnSlot()
        return []
     
    #override   
    def get_out_conn_slots(self):
        # return list of NodeConnSlot()
        return []
            
    #override   
    def serialize(self):
        d = super().serialize()
        #d.update ( {'':self.} )
        return d
    
class FuncParamNode(Node):
    def __init__(self, param_name=None, **kwargs):
        super().__init__(**kwargs)
        self.param_name = param_name
    
    #override
    def get_out_conn_slots(self):        
        return [ NodeConnSlot(self.param_name, NODE_CONN_TYPE.FLOAT ) ]
        
class BoxNode(Node):    
    
    #override   
    def get_in_conn_slots(self):        
        return [ NodeConnSlot("pos", NODE_CONN_TYPE.VEC3 ), NodeConnSlot("size", NODE_CONN_TYPE.VEC3 ) ]
        
    #override
    def get_out_conn_slots(self):        
        return [ NodeConnSlot("dist", NODE_CONN_TYPE.FLOAT ) ]
        
    def place(self, out_str):
        # get conn,
        # f"sdBox ({}, {}) "
        
class RaymarchINNode(Node):
    #override   
    def get_out_conn_slots(self):        
        return [ NodeConnSlot("pos", NODE_CONN_TYPE.VEC3 ) ]
   
class SheetNode(Node):
    """
    nodes container
    """
    def __init__(self, parent_node=None, **kwargs):
        self.parent_node = parent_node
        self.child_nodes = []
        
    def add_node(self, node):
        if node in self.child_nodes:
            raise Exception("already contain node")
        
    def connect_nodes(self, out_node, out_slot_name, in_node, in_node_in_slot_name):
        pass
        
     
class PartSheetNode(SheetNode):
        
    #override   
    def get_in_conn_slots(self):        
        return [ NodeConnSlot("pos", NODE_CONN_TYPE.VEC3 ) ]
   
    #override   
    def get_out_conn_slots(self):        
        return [ NodeConnSlot("dist", NODE_CONN_TYPE.FLOAT), NodeConnSlot("material", NODE_CONN_TYPE.FLOAT)  ]
   
        
          
class ModelSheet(SheetNode):
    """
    Main sheet
    """
    pass
    
class Scene():
    
    def __init__(self):
        self.ms = ModelSheet()
        
    def get_model_sheet(self): return self.ms
    
    def compile(self):
        pass

#import code
#code.interact(local=dict(globals(), **locals()))


"""
mds = ModelSheet()



InputNodes
    RaymarchIN, vec3
    time

MaterialNode
    Name
SolidColorMaterial
    various inputs


PartSheet


rin = RaymarchINNode()
mds.add_node(rin)

rin.connect_to()




PartMaterial


How to group nodes?
NodeGroup( )


"""



    


class GLWidget(QOpenGLWidget):
    
    def __init__(self):
        
        self.data = np.zeros((4, 2), dtype=np.float32)
        self.data[...] = [(-1, -1), (+1, -1), (-1, +1), (+1, +1)]
        super().__init__()
    
        
    def initializeGL(self):
        
        sf = QSurfaceFormat()
        sf.setRenderableType(QSurfaceFormat.OpenGLES)
        self.setFormat(sf)
        
        #self.reinit_shader()
        program = gl.glCreateProgram()
        vertex = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        fragment = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)

        # Set shaders source
        gl.glShaderSource(vertex, vertex_code)
        gl.glShaderSource(fragment, fragment_code )

        # Compile shaders
        gl.glCompileShader(vertex)
        if not gl.glGetShaderiv(vertex, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(vertex).decode()
            logger.error("Vertex shader compilation error: %s", error)

        gl.glCompileShader(fragment)
        if not gl.glGetShaderiv(fragment, gl.GL_COMPILE_STATUS):
            error = gl.glGetShaderInfoLog(fragment).decode()
            print(error)
            raise RuntimeError("Fragment shader compilation error")

        gl.glAttachShader(program, vertex)
        gl.glAttachShader(program, fragment)
        gl.glLinkProgram(program)

        if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
            print(gl.glGetProgramInfoLog(program))
            raise RuntimeError('Linking error')

        gl.glDetachShader(program, vertex)
        gl.glDetachShader(program, fragment)

        gl.glUseProgram(program)

        buffer = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)

        loc = gl.glGetAttribLocation(program, "position")
        gl.glEnableVertexAttribArray(loc)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, buffer)
        gl.glVertexAttribPointer(loc, 2, gl.GL_FLOAT, False, self.data.strides[0], ctypes.c_void_p(0))

        loc = gl.glGetUniformLocation(program, "iResolution")
        gl.glUniform2f(loc, self.size().width(), self.size().height() )

        gl.glBufferData( gl.GL_ARRAY_BUFFER, self.data.nbytes, self.data, gl.GL_DYNAMIC_DRAW)

    def paintGL(self):
        print('paintGL')
        self.initializeGL()
        
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)
        
class MainWindow(QXMainWindow):

    def __init__(self):
        super().__init__()
        
        
        
        scene = self.scene = Scene()
        msn = scene.get_model_sheet()
        
        
        psn = PartSheetNode()
        
        
        
        self.glWidget = GLWidget()

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(self.glWidget)
        
        self.setLayout(mainLayout)
        
        self.resize( QSize(400,400))
        
        self.add_keyPressEvent_listener(self.on_keypress)
        
        self._timer = QTimer()          # creating timer
        self._timer.timeout.connect(self.on_time)
        self._timer.start(1000 / 300)   # setting up timer ticks to 60 fps
    
    def on_time(self):
        self.glWidget.update()            
        
    def paintGL(self):
        pass                                 # some painting code here

      
        
    def on_keypress(self, ev):        
        key = ev.nativeVirtualKey()
        if key == Qt.Key_Space:
            t = time.time()
            
            
            #self.glWidget.initializeGL()
            self.glWidget.update()
            
            print(f'time = {time.time()-t}')
            
            
def main():
    
    root_path = Path(__file__).parent

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication([])
    app.setApplicationName("Test")
    app.setStyle('Fusion')

    #QFontDatabase.addApplicationFont( str(root_path / 'gfx' / 'fonts' / 'NotoSans-Medium.ttf') )
    #app.setFont( QFont('NotoSans'))
    app.setPalette( QDarkPalette() )

    win = MainWindow()

    win.show()
    win.raise_()

    app.exec_()
    
    import code
    code.interact(local=dict(globals(), **locals()))
    
    

if __name__ == "__main__":
    main()
