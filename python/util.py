import numpy
def drawSkeleton(input, hms, coords):
    im = input

    pairRef = (
        (1,2),      (2,3),      (3,7),
        (4,5),      (4,7),      (5,6),
        (7,9),      (9,10),
        (14,9),     (11,12),    (12,13),
        (13,9),     (14,15),    (15,16)
    )

    partNames = ('RAnk','RKne','RHip','LHip','LKne','LAnk',
                       'Pelv','Thrx','Neck','Head',
                       'RWri','RElb','RSho','LSho','LElb','LWri')
    partColor = (1,1,1,2,2,2,0,0,0,0,3,3,3,4,4,4)

    actThresh = 0.002

    # Loop through adjacent joint pairings
    for pair in pairRef:
        if np.mean(hms[pair[0]]) > actThresh and np.mean(hms[pair[1]]) > actThresh:
            # Set appropriate line color
            if partColor[pair[0]] == 1:
              color = (0,85,255)
            elif partColor[pair[0]] == 2 : color = (255,85,0)
            elif partColor[pair[0]] == 3 : color = (0,0,255)
            elif partColor[pair[0]] == 4 : color = (255,0,0)
            else: color = (180,0,180)

            # Draw line
            im = drawLine(im, coords[pair[0]], coords[pair[1]], 4, color, 0)
        
    return im
