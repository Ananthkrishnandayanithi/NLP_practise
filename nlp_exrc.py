# %%
import re

# %%
text = ' Elon mask number is 8989898989 , (732)-666-8888 call him if you have my questions on dodecoin'

# %%
pattern = r'\(\d{3}\)-\d{3}-\d{4}|\d{10}'
re.findall(pattern,text)

# %%
import re

text = ' Elon mask number is 8989898989 , (732)-666-8888 call him if you have my questions on dodecoin'
pattern ='[^;-]'
matches = re.findall(pattern, text)
print(matches)


# %%
import re

text = ' Elon mask number is 8989898989 , (732)-666-8888 call him if you have my questions on dodecoin email him ananth_100@gamil.com '
pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{3}'
matches = re.findall(pattern, text)
print(matches)


# %%
text = "412344 is my order number "
pattern = r'[^a-zA-z]'
matches = re.findall(pattern, text)
print(matches)


# %%
import re

text = "order2 412344 is my order number"
pattern = r'order.*?(\d{6})'
matches = re.findall(pattern, text)
print(matches)


# %%
text = ''' Steven Paul Jobs (February 24, 1955 – October 5, 2011) age-56 was an American businessman, inventor, and investor best known for co-founding the technology company Apple Inc. Jobs was also the founder of NeXT and chairman and majority shareholder of Pixar. He was a pioneer of the personal computer revolution of the 1970s and 1980s, along with his early business partner and fellow Apple co-founder Steve Wozniak.

Jobs was born in San Francisco in 1955 and adopted shortly afterwards. He attended Reed College in 1972 before withdrawing that same year. In 1974, he traveled through India, seeking enlightenment before later studying Zen Buddhism. He and Wozniak co-founded Apple in 1976 to further develop and sell Wozniak's Apple I personal computer. Together, the duo gained fame and wealth a year later with production and sale of the Apple II, one of the first highly successful mass-produced microcomputers.

Jobs saw the commercial potential of the Xerox Alto in 1979, which was mouse-driven and had a graphical user interface (GUI). This led to the development of the largely unsuccessful Apple Lisa in 1983, followed by the breakthrough Macintosh in 1984, the first mass-produced computer with a GUI. The Macintosh launched the desktop publishing industry in 1985 (for example, the Aldus Pagemaker) with the addition of the Apple LaserWriter, the first laser printer to feature vector graphics and PostScript.

In 1985, Jobs departed Apple after a long power struggle with the company's board and its then-CEO, John Sculley. That same year, Jobs took some Apple employees with him to found NeXT, a computer platform development company that specialized in computers for higher-education and business markets, serving as its CEO. In 1986, he helped develop the visual effects industry by funding the computer graphics division of Lucasfilm that eventually spun off independently as Pixar, which produced the first 3D computer-animated feature film Toy Story (1995) and became a leading animation studio, producing 28 films since.

In 1997, Jobs returned to Apple as CEO after the company's acquisition of NeXT. He was largely responsible for reviving Apple, which was on the verge of bankruptcy. He worked closely with British designer Jony Ive to develop a line of products and services that had larger cultural ramifications, beginning with the "Think different" advertising campaign, and leading to the iMac, iTunes, Mac OS X, Apple Store, iPod, iTunes Store, iPhone, App Store, and iPad. Jobs was also a board member at Gap Inc. from 1999 to 2002.[3] In 2003, Jobs was diagnosed with a pancreatic neuroendocrine tumor. He died of tumor-related respiratory arrest in 2011; in 2022, he was posthumously awarded the Presidential Medal of Freedom. Since his death, he has won 141 patents; Jobs holds over 450 patents in total.[4]'''

pattern = r'age-(\d+)'

matches = re.findall(pattern, text)
print(matches)




