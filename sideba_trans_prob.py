import numpy as np
import hashlib
import scipy.special as scsp
import matplotlib.pyplot as plt
import matplotlib.patches as mpat
from matplotlib.backends.backend_pdf import PdfPages

hbar=  6.62607015e-34/(2*np.pi) # J*s
kB=    1.380649e-23 # J/K
Dainkg=1.660539066e-27

class parConf: #Parameter configuration
    def __init__(self,o,ra,wl,m,loadf=True,sr=3,eta=None): #Initialize Parameter configuration (Set eta to override the effect of wl, m & o)
        self.omz=o #Trap frequency
        self.Om0=ra #Rabi frequency
        self.wavelength=wl #Laser wavelength
        if(eta is None):
            self.myeta=2*np.pi/(wl) * np.sqrt(hbar/(2*Dainkg*m*o)) #Lamb-Dicke parameter from wavelength, mass and trap frequency
        else:
            self.myeta=eta #Manual Lamb-Dicke parameter
        self.sran=sr
        idstrenc=(str(o)+str(ra)+str(sr)+str(self.myeta)).encode()
        self.idstr=hashlib.md5(idstrenc).hexdigest() #Parameter configuration id
        self.GLlut=np.empty((0,sr+1), float)
        if(loadf):
            self.loadGL() #Load previously saved data
        self.Omlut=None
        self.normf_gc=lambda omz,T:np.exp(-hbar*omz/(2*kB*T))/(1-np.exp(-hbar*omz/(kB*T))) #norm function (geometric series)
        print(self.myeta,self.idstr[:8])
    def genLine(self,n=1): #Generate GenLaguerre entry
        x=self.myeta**2
        for i in range(n):
            if(self.GLlut.shape[0]==0):
                self.GLlut=np.row_stack( (self.GLlut, np.ones(self.sran+1)) )
            elif(self.GLlut.shape[0]==1):
                apparr=np.ones(self.sran+1)
                for alpha in range(self.sran+1):
                    apparr[alpha]=1+alpha-x
                self.GLlut=np.row_stack( (self.GLlut, apparr) )
            else:
                k=self.GLlut.shape[0]-1
                apparr=np.ones(self.sran+1)
                for alpha in range(self.sran+1):
                    apparr[alpha]=((2*k+1+alpha-x)*self.GLlut[k,alpha]-(k+alpha)*self.GLlut[k-1,alpha])/(k+1)
                self.GLlut=np.row_stack( (self.GLlut, apparr) )
    def saveGL(self): #Save GenLaguerre-buffer
        np.save("data/gl_"+self.idstr[:8]+".npy",self.GLlut)
    def loadGL(self): #Load GenLaguerre-buffer
        try:
            dd=np.load("data/gl_"+self.idstr[:8]+".npy")
        except OSError:
            print("Error loading file for",self.idstr[:8])
            return False
        self.GLlut=dd
        return True
    def saveOmlut(self): #Save Omlut-buffer
        np.save("data/omlut_"+self.idstr[:8]+".npy",self.Omlut)
    def loadOmlut(self): #Load Omlut-buffer
        try:
            dd=np.load("data/omlut_"+self.idstr[:8]+".npy")
        except OSError:
            print("Error loading file for",self.idstr[:8])
            return False
        self.Omlut=dd
        return True
    def facdiv(self,m,n): #Product of integers in range from m to n
        assert m>=0 and n>=0, "One or both parameters negative"
        if(m==0 or n==0):
            if(m!=n):
                return max(m,n)
            else:
                return 1
        o=1
        for i in range(min(m,n)+1,max(m,n)+1):
            o*=i
        return o
    def Om(self,n,s): #Return Rabi frequency
        nle=min(n,n+s)
        nge=max(n,n+s)
        if nle<0:
            return np.nan
        assert self.GLlut.shape[0]>=nle and self.GLlut.shape[1]>=np.abs(s), "GenLaguerre-buffer insufficient"
        nsr=np.sqrt(1./self.facdiv(nle,nge))
        rv=self.Om0*np.exp(-self.myeta**2/2)*self.myeta**(np.abs(s))*nsr*self.GLlut[nle,np.abs(s)]
        return rv
    def genOmlut(self,a,b): #Generate Rabi frequency look-up-table from n=a to n=b
        OmMat=np.empty((0,2*self.sran+1), float)
        for n in range(a,b):
            Oms=np.zeros(2*self.sran+1)
            for s in range(-self.sran,self.sran+1):
                Oms[s+self.sran]=self.Om(n,s)
            OmMat=np.row_stack( (OmMat, Oms) )
        self.Omlut=OmMat
        return OmMat
    def getGrCol(self,s): #Get graph color
        if(s<0):
            return "red"
        elif(s>0):
            return "blue"
        else:
            return "black"
    def getLineSt(self,s): #Get Linestyle
        sa=min(abs(s),3)
        lss=["-","--",":","-."]
        return lss[sa]
    def P_lut(self,T,s,t,scale_t=False,aC=0.01): #Transition Probability look up table
        assert (self.Omlut is not None), "Omega look-up-table not loaded"
        PB=lambda n,T:np.exp(-hbar*self.omz*(n+1/2)/(kB*T)) #Boltzmann Probability
        Pc_lut=lambda n,T,s,t:PB(n,T)*np.sin(self.Omlut[n,s+self.sran]*t/2)**2
        
        if(scale_t):
            t*=np.pi/(self.myeta*self.Om0)
        o=0
        norm=self.normf_gc(self.omz,T)
        nsm=int(np.ceil( -np.log(aC)*(kB*T)/(hbar*self.omz) ))
        assert self.Omlut.shape[0]>=nsm, "Omega look-up-table insufficient (needs "+str(nsm)+" has "+str(self.Omlut.shape[0])+")"
        for n in range(max(0,-s),nsm):
            o+=Pc_lut(n,T,s,t)
        return o/norm
    def dP_inv4(self,C,a,k=1,mx0=10): #Function for estimation of max n (in sum for dP/dT)
        dP_inv3=lambda y,a,k:-scsp.lambertw(y,k)/a-1
        rt=[]
        myInv=np.real(dP_inv3(-C,a,k))
        if myInv.size<2:
            return myInv if myInv>mx0 else mx0
        for e in myInv:
            rt.append( e if e>mx0 else mx0 )
        return np.array(rt)
    def T_dPdT_lut(self,T,s,t,scale_t=False,aC=0.01,nsm=200): #dP/dT look-up-table
        T_dPdTc_lut=lambda n,T,s,t:np.exp(-hbar*self.omz*n/(kB*T))*(n-(n+1)*np.exp(-hbar*self.omz/(kB*T)))*np.sin(self.Omlut[n,s+self.sran]*t/2)**2
        o=0
        if(scale_t):
            t*=np.pi/(self.myeta*self.Om0)
        #norm=hbar*self.omz/(kB*T**2)
        alfa=hbar*self.omz/(kB*T)
        nsmr=np.real(  self.dP_inv4(aC,alfa,1,5)  )
        nsm=1
        if not np.isnan(nsmr):
            nsm=int(np.ceil(nsmr))
        for n in range(max(0,-s),nsm):
            o+=T_dPdTc_lut(n,T,s,t)
        return o*alfa
    def myLegendAxis(self,imax,txt): #Make custom legend
        tsize=8
        #imax = pf.add_axes([.45,.5,.25,.25])
        legbb=mpat.FancyBboxPatch((0,0),5,self.sran/2,facecolor=[1,1,1,.8],edgecolor="#cacacacc",boxstyle="Round, pad=.5")
        imax.add_patch(legbb)
        for i in range(1,self.sran+1):
            imax.hlines(i/2,0,1,ls=self.getLineSt(i),color=self.getGrCol(i))
            imax.text(1.25,i/2,"s="+str(i),va='center',size=tsize)
            imax.hlines(i/2,2.5,3.5,ls=self.getLineSt(i),color=self.getGrCol(-i))
            imax.text(3.75,i/2,"s=-"+str(i),va='center',size=tsize)
        imax.hlines(0,2.5,3.5,ls=self.getLineSt(0),color=self.getGrCol(0))
        imax.text(3.75,0,"s=0",va='center',size=tsize)
        imax.hlines(0,0,.6,color="#505050",linewidth=.8)
        imax.text(1.25-.4,0,txt,va='center',size=tsize)
        imax.axis('off')
    def doublePlot(self,per,titlinfo="",C_P=0.01,C_dPdT=0.001):
        logxs=np.linspace(-6,-2,129)
        xs=10**logxs

        mysran=range(-self.sran,self.sran+1)
        pf,ax=plt.subplots(2,1,figsize=(8*2**(1/2),8))
        pratio=per.as_integer_ratio()
        mymat=[]
        for x,lx in zip(xs,logxs):
            one_s=[]
            for mys in mysran:
                one_s.append(self.P_lut(x,mys,per,True,C_P))
            mymat.append(one_s)
        mymat=np.array(mymat)
        for mm,sss in zip(np.array(mymat).T,mysran):
            myc=self.getGrCol(sss)
            myls=self.getLineSt(sss)
            ax[0].plot(xs,mm,label="s = "+str(sss),c=myc,ls=myls)
        ax[0].hlines(1/2,1e-6,1e-2,color="#505050",linewidth=.8,label="P = 1/2")
        ax[0].set_title("$\\frac{"+str(pratio[0])+"}{"+str(pratio[1])+"}\\pi$-pulse, n=0, $P_{≥n_{max}}$="+str(C_P)+", "+titlinfo+"\n$\\omega_z=2\\pi\\cdot$"+str(self.omz/(2*np.pi*1e3))+" kHz, $\\Omega_0=2\\pi\\cdot$"+str(self.Om0/(2*np.pi*1e3))+" kHz, $\\lambda=$"+str(self.wavelength*1e9)+" nm")
        ax[0].set_ylim([0,1])
        ax[0].set_ylabel("Transition probability")
        
        mymat=[]
        for x,lx in zip(xs,logxs):
            one_s=[]
            for mys in mysran:
                one_s.append(self.T_dPdT_lut(x,mys,per,True,C_dPdT))
            mymat.append(one_s)
        mymat=np.array(mymat)
        for mm,sss in zip(np.array(mymat).T,mysran):
            myc=self.getGrCol(sss)
            myls=self.getLineSt(sss)
            ax[1].plot(xs,mm,label=sss,c=myc,ls=myls)
        ax[1].hlines(0,1e-6,1e-2,color="#505050",linewidth=.8,label="T$\\cdot$dP/dT = 0")
        ax[1].set_title("$\\frac{"+str(pratio[0])+"}{"+str(pratio[1])+"}\\pi$-pulse, n=0, $\\mathcal{C}$=-"+str(C_dPdT)+", "+titlinfo+"\n$\\omega_z=2\\pi\\cdot$"+str(self.omz/(2*np.pi*1e3))+" kHz, $\\Omega_0=2\\pi\\cdot$"+str(self.Om0/(2*np.pi*1e3))+" kHz, $\\lambda=$"+str(self.wavelength*1e9)+" nm")
        ax[1].set_ylim([-.35,.35])
        ax[1].set_ylabel("T$\\cdot$dP/dT (1)")
        
        imax = pf.add_axes([.85,.59,.15,.15])
        self.myLegendAxis(imax,"P=$\\frac{1}{2}$")
        imax2 = pf.add_axes([.85,.09,.15,.15])
        self.myLegendAxis(imax2,"T$\\cdot\\frac{dP}{dT}$=0")
        
        for a in ax:
            a.set_xscale("log")
            a.set_xlabel("T (K)")
            a.set_xlim([6.5e-7,0.5e-1])
        pf.tight_layout()
        return pf,ax

#Test program below. Delete or play around with it
generate_GLs=False
generate_Omluts=False
export_to_pdf=False
perarr=np.append([0,1/16,1/8,1/4],np.linspace(1/2,2,7))

for fr in [70e3,100e3,150e3,250e3,500e3,1000e3,2000e3]:
    a=parConf(2*np.pi*fr,2*np.pi*16e3,1762e-9,138,True)
    if(generate_GLs):
        a.genLine(30000) #this line takes a while
        a.saveGL() #cache results for later
    if(generate_Omluts):
        a.genOmlut(0,30000)
        a.saveOmlut()
    a.loadOmlut()
    print(fr,a.GLlut.shape)
    if(export_to_pdf):
        with PdfPages('plots/total_plot(T,s,t)_'+str(fr/1e3)+'.pdf') as pdf:
            for per in perarr:
                pf,pa=a.doublePlot(per,"${}^{138}$Ba${}^+$")
                pdf.savefig()
                plt.close()
            d = pdf.infodict()
            d['Author'] = "Lazy GitHub user"
            d['Title'] = "Sideband transition probabilities of trapped ions"
a=parConf(2*np.pi*70e3,2*np.pi*16e3,1762e-9,138,True,3,0.082)
#a=parConf(2*np.pi*2000e3,2*np.pi*16e3,1762e-9,138,True)
#a.genLine(300000)
#a.saveGL()
#a.genOmlut(0,30000)
#print(a.GLlut.shape)