import Image, ImageOps
from glob import glob


#box = (1938, 278, 3594, 968)
#box = (664, 61, 1251, 183)
files =  glob("graphs/alpha*.png")

print len(files)

#x, y = (577 + 50, 122 + 10)
#x, y = (388, 300-26+5)
x, y = (800, 600)

for i, f in enumerate(files):
        im1 = Image.open(f)
        new = Image.new('RGB', (x/2, y/2))
        new.paste( im1, (0, 0) )
        new.save('Rapport/img/%i.png' % i)        

