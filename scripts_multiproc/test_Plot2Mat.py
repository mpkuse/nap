import numpy as np
import cv2

from Plot2Mat import Plot2Mat

y = np.random.random( 1000 )
# y = np.zeros( 20 ) + .2
print y
obj = Plot2Mat()
im = obj.plot( y )
cv2.imshow( 'im', im )
cv2.waitKey(0)
