"""
Algorithms for pre and post processing and analysis of glacier and climate data 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy as car
import netCDF4 as nc 
from itertools import compress
import glob
import os
from scipy import stats as st





#FOR WHEN REFERING TO DICTS, HAVE HTAT AS A VARIABLE THAT REFERS TO THESE DICTS; IF U WANT, YOU CAN SET YOUR OWN

#NEED TO ADD DEBUG AND VERBOSE OPTIONS!! 


###########################################################
# PLAYGROUND
# fp=sorted(glob.glob(os.getcwd()+'/Project_Data/Model_Input_Data/Hypsometry_Data/RGI/00_rgi60_attribs/*.csv'))
# # df=pd.read_csv(fp[0])
#era_temp=nc.Dataset(os.getcwd()+'/Project_Data/Model_Input_Data/Climate_Data/ERA_Int/ERAInterim_Temp.nc')

#reanalysis_map(era_temp,'t2m',latbound=[90,50],timestat=[9,12], yearbound=[1990,2017, 1979])


#temp=pd.read_csv(os.getcwd()+'/Project_Data/Model_Input_Data/Climate_Data/ERAInt_Sim_Selection/RGI_01_ERA_Int_Glacier_Temp.csv').iloc[:,1:]
#ppt=pd.read_csv(os.getcwd()+'/Project_Data/Model_Input_Data/Climate_Data/ERAInt_Sim_Selection/RGI_01_ERA_Int_Glacier_PPT.csv').iloc[:,1:]
 
#climate_plots(temp,ppt,1, yearlim=[2000,2010], monthlim=[3,7], templim=[-20,14],preclim=[0,3] )

# for count, x in enumerate([1,3,4,6,7,8,9]): 
#       df=pd.read_csv(fp[x-1], encoding='latin')
#       area_map(df,x,areaname='Area')


#glac_fp=os.getcwd()+'/Project_Data/Model_Input_Data/Glacier_Data/WGMS_2019'

#geo_fp=os.getcwd()+'/Project_Data/Model_Input_Data/Hypsometry_Data/RGI'

#test_fp=sorted(glob.glob(os.getcwd()+'/Project_Data/Calibration_Outputs/*.nc'))

#valid_fp=sorted(glob.glob(os.getcwd()+'/Project_Data/Calibration_Outputs/ZEMP_Data/*.csv'))

#dat_fp=sorted(glob.glob(os.getcwd()+'/Project_Data/Modelled_Data/*.nc'))

#dat_fp=sorted(glob.glob(os.getcwd()+'/Project_Data/Model_Input_Data/Hypsometry_Data/RGI/00_rgi60_attribs/*.csv'))

#dat_fp=sorted(glob.glob(os.getcwd()+'/Project_Data/Model_Input_Data/Hypsometry_Data/IceThickness/bands_10/*.csv'))

##########################################

#initial functions/setups
def rgi_num(num):
    """
    rgi_num takes the numerical rgi number int and makes it into a string of len=2
    
    INPUT: 
    
    num (int): rgi number to be converted
    
    OUTPUT: 
    
    string
    
    """
    
    if len(str(num))==1:
        num='0'+str(num)
    else: 
        num=str(num) 
    return num


region_dict={
    1:['Alaska',86725],
    2:['Western Canada/US',14524],
    3:['Canada North',105111],
    4:['Canada South',40888],
    5:['Greenland',89680],
    6:['Iceland',11060],
    7:['Svalbard',33959],
    8:['Scandinavia',2949],
    9:['Russian Arctic',51592],
    10:['North Asia',2325],
    11:['Central Europe',2092],
    12:['Middle East',1151],
    13:['Central Asia',49303],
    14:['South Asia West',33568],
    15:['South Asia East',14734],
    16:['Low Latitudes',2341],
    17:['South Andes',29428],
    18:['New Zealand',1161],
    19:['Antarctic',124040]
}

colorleg={
    'glacier':'#4896CB',
    'temp':'r',
    'ppt':'b',
    'massbal':'#9C1313',
    'neutral':'#959595',
    'geo':'#E00AE7',
    'glac':'#976DC3',
    'valid':'#B4950B',
    'melt':'#AC5E85',
    'acc':'#4897DE',
    'refreeze':'#48DE9E',
    'frontalablation':'#E1A510',
    'T0': '#C6B5AC',
    'T1':'#C8957C',
    'T2':'#CD7042',
    'T3':'#CF4804'
    
}


reanalysis={
    't2m':['2m_Air_Temperature','C','coolwarm'],
    'tp':['Annual_Precipitation','M','Blues'],
    'lapserate':['Temperature_Lapse_Rate','C˚m$^{-1}$','reds']
    }

#funct for all latlong maps

#TO DO: 
# 2. have option to determine WHERE the sumstats are located 


def area_map(df, region, latname='CenLat', longname='CenLon', areaname=10, fig_size=(10,20), 
             proj=ccrs.Mercator(), ticks=None, sum_stat=1, plotbase=False,fp=None, boundbox=None,
             save_fig=False, savefp=None,filename='GlacierMap'):
    
    

    """
    area_map maps the location of glaciers in the distinct RGI Regions 
    
    INPUT: 
    
    df (dataframe): dataframe from which the data shall be mapped; requires columns for at least 
        latitude, longitude, and region number 
        
    region (int): an intiger indicating the RGI glacier region (1-19) that is to be mapped.
                    
    latname (str): name of the column containing latitude
    
    longname (str): name of the column contaning longitude
    
    areaname (int or str): if an 'int' is passed, the size of data points representing the glaciers 
        in the reigon if 'str' the name of the column containing the glacier areas
        
    fig_size(tuple, 2 floats): a tuple containing the (width, height) for fig size. Default is (10,20). 
    
    proj (ccrs projection object): custom projection as a ccrs object. Default is ccrs.Mercator().
        !only rectangular ccordinate systems supported for this function!
    
    ticks (tuple, 2 int): a tuple containing the (xtick,ytick) distribution for long/lat coordinates 
        (i.e. a tuple of (3,2) would mean that every third longitdue and every second latitude coordinate would 
        print as ticks on the axes). If 'None', tick values are customaized based on what is optimal for the region. 
    
    sum_stat (int): an int indicating whether summary statistics should be printed. If sum_stat=0, no stats are printed 
        if sum_stat=1, total glacier area is printed, if sum_stat=2, total galcier area, glacier area containing data, and 
        % glacier area containing data are printed. Option 2 is intended to wrok with plotbase. Default is 1. 
        
    plotbase (boolean): indicate whether a 'baseplot' of all the glaciers in the region should be printed. Useful 
        for analysis/comparison of glaciers that have data vs ones that don't. Default is False. 
    
    fp (str): filepath in which to find the RGI data files to plot the 'plotbase'.
    
    boundbox (list, 4 int): a bounding box of [min long, max long, min lat, max lat] you want to display on your map. All sumstats 
        will also be constrained by this. The default is None, meaning that the min/max lat/long will be constrained 
        by the max/min lat/long of the glaciers in the region 
    
    
    save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 
    
    savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
        current file location. Default is None. 
        
    filename (str): what the saved figure should be named. Default is 'GlacierMap'


    OUTPUT: 
        Scatter plot figure
    """
    
    #determine the 2 digit region number 
    num=rgi_num(region)
    
    #define how land mass/brorders will be represented on map 
    land_50m = car.feature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='k',
                                    facecolor='none')
    borders_50m= car.feature.NaturalEarthFeature('cultural', 'admin_0_boundary_lines_land','50m',
                                                 edgecolor='k',
                                                 facecolor='none')

    #define point projection 
    proj_crs = ccrs.PlateCarree()

    #create objects so that lat/long coordinates can be represented with projection
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    
    # constrain the latitude and longitude as per the bounding box specification 
    if boundbox is not None: 
        df=df[(df[latname]>=boundbox[2])&(df[latname]<=boundbox[3]) &
              (df[longname]>=boundbox[0])&(df[longname]<=boundbox[1])]
    
    #extract lat and long names 
    lat=df[latname]
    long=df[longname]
    
    #extract how area will be represented: if string, extract the series of area variables 
    if type(areaname) ==str: 
        
        area=df[areaname]
    #otherwise assign the intiger 
    else: 
        area=areaname
    
    #create the n/s/e/w location points for plot
        
    east = int(round(long.min())) - 1
    west = int(round(long.max())) + 1
    south = int(round(lat.min())) - 1
    north = int(round(lat.max())) + 1

        
    
    #if size to be represented by plot, magnitude of each area is x2 for better visualization 
    #10x in Scandinavia due to small size of glaciers 
    mag=1
    if type(areaname) ==str: 
        mag=1.5
    elif type(areaname) ==str and region==8:
        mag=10
        
    #specify spread of x and y ticks on plot, if None, optimize 
    if ticks is None: 
        if region == 6: 
            xtick=2
            ytick=1
        elif region ==8:
            xtick=5
            ytick=2
        else:
            xtick=8
            ytick=2
    #if assigned, extract as per assigned values 
    else: 
        xtick=ticks[0]
        ytick=ticks[1]
        
        
    fig, axs=plt.subplots(1,1, figsize=fig_size, subplot_kw={'projection': proj})
    #start of plotting, starting with multiple regions, which use 'ravel' function    

    axs.add_feature(land_50m)
    axs.add_feature(borders_50m)

    #option to add plotbase 
    if plotbase==True: 
        # if fp is none, assume data are organized as per the following file 
        if fp is None: 
            fp=sorted(glob.glob(os.getcwd()+'/Project_Data/Model_Input_Data/Hypsometry_Data/RGI/00_rgi60_attribs/*.csv'))
        #extract file 
        temp=pd.read_csv([y for y in fp if num + '_rgi60' in y][0], encoding='latin')
        
        #since this is a new dataset, will once again have to determine the parameters of the bounding box 
        if boundbox is not None: 
            temp=temp[(temp['CenLat']>=boundbox[2])&(temp['CenLat']<=boundbox[3]) &
              (df['CenLon']>=boundbox[0])&(df['CenLon']<=boundbox[1])]
        
        #create the plot 
        axs.scatter(temp['CenLon'],temp['CenLat'], transform=proj_crs, 
                       s=temp['Area']*mag, alpha=0.7, c=colorleg['neutral'])

      
        #this is needed in order to calculate the area properly within a bounding box for sum stat 2 
        temp_area=np.sum(temp['Area'])/1000

        #recreate the n/s/e/w coordinate bounds with plotbase; this will override the prior assignment 
        east = int(round(temp['CenLon'].min())) - 1
        west = int(round(temp['CenLon'].max())) + 1
        south = int(round(temp['CenLat'].min())) - 1
        north = int(round(temp['CenLat'].max())) + 1

    #plot glacier data of interest 
    axs.scatter(long,lat, transform=proj_crs, 
                               s=area*mag, alpha=0.7, edgecolor='black', c=colorleg['glacier'])
    #set ticks as per assigned bounding box 
    axs.set_xticks(np.arange(round(east),round(west)+1,xtick), crs=proj_crs)
    axs.set_yticks(np.arange(round(south),round(north)+1,ytick), crs=proj_crs)
    axs.xaxis.set_major_formatter(lon_formatter)
    axs.yaxis.set_major_formatter(lat_formatter)
    
    #do summary statistics 
    if sum_stat==1: 
        #if area of glaciers is not provided in inputs, will use the default provided by the RGI. This cannot be constrained by bounding box 
        if type(areaname) !=str: 
            area_col=df.columns[df.columns.str.lower().str.contains('area')][0]
            area=np.sum(df[area_col])/1000
            #alert that this is how area is calculated 
            print('Total glacier area calculated using column: ' + area_col)
        #otherwise, calculate glacier area 
        else:
            area=np.sum(area)/1000
        #print values 
        axs.text(0.01,0.96,'Glac Area: '+str(round(area,2))+' x10$^{3}$km$^{2}$',transform=axs.transAxes, fontsize=18)
        axs.text(0.01,0.90,'No. Glac: '+str(len(df)), transform=axs.transAxes, fontsize=18)
    
    #second summary statistics 
    elif sum_stat==2: 
        #if area name is not provided, try to search for it in column names 
        if type(areaname) !=str: 
            area_col=df.columns[df.columns.str.lower().str.contains('area')][0]
            area=np.sum(df[area_col])/1000
            print('observational glacier area calculated using column: ' + area_col)
         
        #if atotal area, will use the default provided by the RGI. This cannot be constrained by bounding box 
        try: 
            total_area=temp_area
        except: 
            total_area=region_dict[region][1]/1000
            print('total glacier area + stats calculated using regional total, please turn on "plot_base" if want to find area within bounding box')
        
        #print statistics 
        axs.text(0.01,.96,'area with data:' + str(round(area,2)) +'x10$^{3}$ km$^{2}$',
                horizontalalignment='left',transform=axs.transAxes, fontsize=15)
        axs.text(0.01,.90,'total glac area:' + str(round(total_area,2)) +'x10$^{3}$ km$^{2}$',
                horizontalalignment='left',transform=axs.transAxes, fontsize=15)
        axs.text(0.01,.84,'% area with data:' + str(round((area/total_area)*100,1)) +' %',
                horizontalalignment='left',transform=axs.transAxes, fontsize=15)
     
    #set title
    axs.set_title(region_dict[region][0],fontdict={'fontsize':20})
    
    
    #option to save figure 
    if save_fig==True: 
        if savefp is None: 
            plt.savefig(os.getcwd()+'/R_'+num+'_'+filename+'.png')
        else:
            plt.savefig(savefp+'/R'+num+'_'+filename+'.png')


#ERA maps

def reanalysis_map(data_file, dat_type, fig_size=(20,15),latbound=[90,-90], longbound=[180,-180], timestat=[1,12], yearbound=None,
                    map_title=None, units=None,colmap=None, save_fig=False,savefp=None, filename='ERA_Plot'):
    
    #yearbound a list of 3 numbers: startyear, endyear, that u want, and startyear of dataset
    
    """
    reanalysis_map maps the data from a climate reanalysis prodcut for visualization 
    
    INPUT: 
        
    data_file (NetCDF File): a NetCDF file containing the variable of interest in monthly resolution.
    
    dat_type (str): name of the variable, as it appears on the NetCDF file 
    
    fig_size(tuple, 2 floats): a tuple containing the (width, height) for fig size. Default is (20,15). 
    
    latbound (list, 2 int): list of [north, south] latitude (in decimal degrees) that should 
        be displayed. NetCDF file must extend to these bounds. Default is [90,-90] (full globe).
        
    longbound (list, 2 int): list of [east, west] longitude (in decimal degrees) that should
        be displayed. NetCDF file must extend to these bounds. Default is [180,-180] (full globe). 
        
    timestat (list, 2 int): the range of months to be included in calculation. The default is all 12 months. 
    
    yearbound (list, 3 int): the range of years to be included in calculation, as well as the start year of the
        dataset: [startyear, endyear, dataset startyear]. 
        
    map_title (str): title of map. If 'None', it 'None', it is assigned a pre-set string 
        (available for temp, ppt, and lapse rate)
    
    units (str): units of measured variable. If 'None', it is assigned a pre-set string 
        (available for temp, ppt, and lapse rate)
        
    colmap (str): name of colormap to use for the plot. If 'None', it is assigned a pre-set string 
        (available for temp, ppt, and lapse rate)
          
    save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 
    
    savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
        current file location. Default is None. 
        
    filename (str): what the saved figure should be named. Default is 'ERA_Plot'
    
    
    OUTPUT: 
        contour plot figure
    """
    
    
    #extract the latitude, longitude and variable of interest
    latitude=data_file['latitude']
    longitude=data_file['longitude']
    temp=data_file[dat_type]
    
    #identify lat/long based on the latitude and longitude user-specified bounds 
    #era-int lat/long are given in 180/360 degree format rather than 90/180, at various resolutions, so need to convert 
    northlat=int(round((90-latbound[0])*(len(latitude)/180)))
    southlat=int(round((90-latbound[1])*(len(latitude)/180)))
    eastlong=int(round((180-longbound[0])*(len(longitude)/360)))
    westlong=int(round((180-longbound[1])*(len(longitude)/360)))
    
    #bind based on identified lat/long
    latitude=latitude[northlat:southlat]
    longitude=longitude[eastlong:westlong]
    
    #identify which years are to be calculated, using the identified start year to convert from the year to its location in the file 
    if yearbound is not None: 
        startyear=int(yearbound[0]-yearbound[2])*12
        endyear=int(yearbound[1]-yearbound[2])*12
        temp=temp[startyear:endyear,northlat:southlat,eastlong:westlong]
    #otherwise, just bind by lat/long 
    else: 
        temp=temp[:,northlat:southlat,eastlong:westlong]

    #for the range of months user is interested in, create yearly average value 
    #if data is precipitation (tp) create sum rather than mean (as ppt is measured as an annual cumilative value)
    data_set=[]
    if dat_type=='tp':
        for x in list(range(0,int(len(temp)/12))): 
            #we want to pull values 0-11 over an interval of 12 
            #the user input is for months rather than the index values which are being pulled
            #we need to subtract by 1 to pull the index of interest for the startmonth
            #however, because python only *pulls up* to the sliced value, the endmonth can be left alone 
            #i.e. [5:12] will pull for index 5 to 11
            data_set.append(np.sum(temp[(x*12)+(timestat[0]-1):(x*12)+timestat[1],:,:], axis=0))
    else: 
        for x in list(range(0,int(len(temp)/12))): 
            data_set.append(np.mean(temp[(x*12)+timestat[0]:(x*12)+timestat[1],:,:], axis=0))
     
    #if map/units/colormap were not user specified, first try the preset values for temperature/precipitation/lapse rate 
    #if the variable is not on our preset list, put generic titles to indicate to user that these need to be set 
    if map_title is None: 
        try: 
            map_title=reanalysis[dat_type][0].replace('_',' ')
        except: 
            map_title='MEASUREMENT'
    if units is None: 
        try:
            units=reanalysis[dat_type][1]
        except: 
            units='UNITS'
    if colmap is None: 
        try:
            colmap=reanalysis[dat_type][2]
        except: 
            colmap='grey'
    
    #create a mean value across all years of interest     
    snap=np.mean(data_set, axis=0)

    #set land mass for plotting purposes 
    land_50m = car.feature.NaturalEarthFeature('physical', 'land', '50m',
                                    edgecolor='k',
                                    facecolor='none')

    #set figure, projection, and add land as a feature 
    fig, ax=plt.subplots(figsize=fig_size)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(land_50m)
    #contor plots to map variables 
    cs=ax.contourf(longitude, latitude, snap,20, cmap= colmap)
    #set colorbar and map titles 
    cb = fig.colorbar(cs, fraction=0.02, pad=0.04, orientation='horizontal')
    plt.title(map_title)
    cb.set_label(map_title+'_'+units)
    
      
    #save figure 
    if save_fig==True: 
        if filename=='ERA_Plot':
            filename=filename+'_mean_'+map_title
        if savefp is None:
            plt.savefig(os.getcwd()+'/'+filename+'.png')
        else:
            plt.savefig('/'+savefp+filename+'.png')
    fig.show()



#temp ppt plots 

def climate_plots(temp_file, ppt_file, region, fig_size=(9,5),yearstart=2000, yearlim=None, monthlim=None, 
                  templim=[-40,15],preclim=[0,2.5],sum_stats=True,save_fig=False, savefp=None, figname=None):
    
    
    """
    climate_plots displays the annual and mean monthly temperature and precipitation in the regions of interest
    
    INPUT: 
        
    temp_file (dataframe): dataframe with the annual temperature values for all glaciers in a single RGI region
    
    ppt_file (dataframe): dataframe with annual precipitation values for all glaciers in a single RGI region 
    
    region (int): an intiger to indicate the RGI region number (1-19) that is to be plotted 
    
    fig_size(tuple, 2 floats): a tuple containing the (width, height) for fig size. Default is (9,5). 
    
    startyear (int): an intiger to indicate the startyear of the dataset. Default is 2000. 
    
    yearlim (list, 2 int): a list of 2 intigers to indicate the desired start and end years for analysis. 
        If 'None', analysis will be conducted on full dataset. Default is 'None'. 
        
    monthlim (list, 2 int): a list of 2 integers to indicate the desired start and end months for analysis, 
        in 'water years' (October to September). If 'None', full water year analyzed. Default is 'None'. 
        
    templim (list, 2 float): a list of 2 floats to indicate the y-lim for the temperature plots. Default is [-40, 15]. 
    
    preclim (list, 2 float): a list of 2 floats to indicate y-lim for the annual precipitation plots. 
        This value is divided by 8 for mean monthly plots. Default is [0, 2.5]. 
        
    sum_stats (boolean): boolean value to indicate whether or not to display summary statistics. Default is 'True'. 
    
    save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 
    
    savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
        current file location. Default is None. 
        
    filename (str): what the saved figure should be named. Default is 'R[region number]_climate_summary'.
    
    OUTPUT: 
        plot figure 
    """
    
    #create a datelist for the plots 
    datelist=['Oct','Nov','Dec','Jan','Feb','March','April','May','Jun','Jul','Aug','Sept']

    #calculate annual and monthly mean temp for glaicers across total time period 
    annual_temp=np.nanmean((np.array(np.nanmean(temp_file,axis=0)).reshape(-1,12)),axis=1)
    monthmean_temp=np.nanmean((np.array(np.mean(temp_file,axis=0)).reshape(-1,12)),axis=0)
    
    #calculate annual total and monthly total ppt for a single glacier 
    annual_ppt=np.nansum((np.array(np.nanmean(ppt_file,axis=0)).reshape(-1,12)),axis=1)
    monthsum_ppt=np.nanmean((np.array(np.nanmean(ppt_file,axis=0)).reshape(-1,12)),axis=0)

    #create a list of years available in the file 
    year=list(range(yearstart, yearstart+len(annual_temp)))
    
    #constrain years based on user specification 
    if yearlim is not None: 
        year=list(range(yearlim[0],yearlim[1]))
        annual_temp=annual_temp[yearlim[0]-yearstart:yearlim[1]-yearstart]
        annual_ppt=annual_ppt[yearlim[0]-yearstart:yearlim[1]-yearstart]
    
    
    #constrain months based on user specification. Because we are working in water years, need to convert user-speified years to match index
    if monthlim is not None: 
        
        #if month is oct, nov, dec (beginning of water year), adjust to beggining of dataset
        if monthlim[0] >9: 
            adj_1=-10
        elif monthlim[0]<10: 
            adj_1=2
            
        if monthlim[1]>9: 
            adj_2=-10
        elif monthlim[1]<10: 
            adj_2=2
        
        
        monthlim=[monthlim[0]+adj_1,monthlim[1]+adj_2]
        monthmean_temp=monthmean_temp[monthlim[0]:monthlim[1]]
        monthsum_ppt=monthsum_ppt[monthlim[0]:monthlim[1]]
    else: 
        monthlim=[0,12]
        
    #create a figure with 2 plots, one for monthly and one for annual summaries 
    fig, (ax,ax3)= plt.subplots(2,1, figsize=fig_size)

    #create a reference line at which point temp is 0 C
    ax.axhline(0,ls='--',c=colorleg['neutral'],alpha=0.5)
    ax3.axhline(0,ls='--',c=colorleg['neutral'], alpha=0.5)
    ax.plot(year,annual_temp, c=colorleg['temp'])
    ax.set_title(region_dict[region][0])
    ax.set_ylabel('Mean Annual Temp (˚C)', c=colorleg['temp'])
    #plot precip on a secondary y axis for both plots 
    ax2=ax.twinx()
    ax2.plot(year,annual_ppt, c=colorleg['ppt'])
    ax2.set_ylabel('Mean Annual PPT (m)',c=colorleg['ppt'],rotation=270)
    ax4=ax3.twinx()
    ax4.plot(monthsum_ppt,c=colorleg['ppt'])
    ax3.plot(monthmean_temp, c=colorleg['temp'])
    if len(monthmean_temp)>5: 
        monthrange=list(range(0,len(monthmean_temp),2))
        monthskip=2
    else: 
        monthrange=list(range(len(monthmean_temp)))
        monthskip=1
    ax3.set_xticks(monthrange)
    ax3.set_xticklabels([datelist[t] for t in list(range(monthlim[0],monthlim[1],monthskip))],
                                      rotation = 45, ha="right")
    ax3.set_ylabel('Mean Monthly Temp (˚C)',c=colorleg['temp'])
    ax4.set_ylabel('Mean Monthly PPT (m)',c=colorleg['ppt'],rotation=270)
    ax4.yaxis.set_label_coords(1.085,0.5)
    ax2.yaxis.set_label_coords(1.085,0.5)
    #setting a universal y limit for both temp and precipitation so plots are comparable against each other 
    ax.set_ylim(templim[0],templim[1])
    ax3.set_ylim(templim[0],templim[1])
    ax2.set_ylim([preclim[0],preclim[1]])
    #divide ylim for mean monthly visualization
    ax4.set_ylim([preclim[0]/8,preclim[1]/8])
    #fill precip plots when temp is >0 to indicate that ppt is solid at the time 
    ax4.fill_between(list(range(len(monthsum_ppt))),monthsum_ppt, where=np.array(monthmean_temp<0), color='b',alpha=0.2)
    
    if sum_stats==True: 
        #create summary stats for each plot 
        ax4.text(1.13,0.93,'mean annual temperature: ' + str(round(np.mean(monthmean_temp),1)) + ' ˚C', 
                  horizontalalignment='left', transform=ax4.transAxes)
        ax4.text(1.13,0.83,'total solid precipitation: ' + str(round(np.sum(monthsum_ppt[monthmean_temp<0]),3)) + ' m', 
                  horizontalalignment='left', transform=ax4.transAxes)
        ax4.text(1.13,0.73,'max temperature amplitude: ' + str(round(np.max(monthmean_temp)-np.min(monthmean_temp),1)) + ' ˚C', 
                  horizontalalignment='left', transform=ax4.transAxes)
        
    #save file 
    if save_fig==True: 
        num=rgi_num(region)
        if figname is None: 
            figname='R'+num+'_climate_summary'
        if savefp is None:
            plt.savefig(os.getcwd()+'/'+figname+'.png')
        else:
            plt.savefig(savefp+'/'+figname+'.png')   
    
    plt.show()

#process: massbal obs data 
    
   #TO DO: 
# 3. have option to determine WHERE the legend located (need to think about how legend comes about bc not all regions have all data types)
# 6. may have to figure out the del axis thing... if we need to remove more than one plot 
class obs_massbal:
    def __init__(self, wgms_fp, rgi_fp,  preprocess=True, glac_fin=None):
        
        """
        the class obs_massbal can process all available mass balance data from the WGMS and convert it into 
        annual mass balance (in m w.e.) and attach the values to RGI id numbers. 
        
        INPUT: 
            
        wgms_fp (str or dataframe): a string that leads to the location of the wgms glaciological and geodetic mass balance data 
            if preprocess=True and data have already been preprocessed, this is the dataframe of all available mass balance 
            data that have already been preprocessed 
        
        rgi_fp (str or dataframe): a string that leads to the location of the RGI dataset. If preprocess=True and the data have already 
            been preprocessed, this is the dataframe of geodetic mass balance values available 
        
        preprocess (boolean): an indication whether the the input data need to be preprocessed or not. Default is 'True' 
            (therefore need to be preprocessed)
            
        glac_fin (dataframe): when preprocess=True, this is the datagrame of the preprocessed glaciological mass balance data 
        
        OUTPUT: 
            initialized class object 
        """
        
        self.wgms_fp=wgms_fp
        self.rgi_fp=rgi_fp
        self.preprocess=preprocess
        self.glac_fin=glac_fin
        #get filepaths where all wgsm data are stored and where all rgi data are stored, not just the files
        
    def massbal_process(self, regions, yearstart=None,yearend=None):

        """
        massbal_process processes the data input to initialize obs_massbal object, if preprocess=True, this 
        function simply identifies and assignes the variables into their appropriate spaces 
        
        INPUT: 
            regions (list of n int): list of intigers indicating which RGI regions should be looked at.
            
            yearstart (int): an intiger indicating which year analysis should be started. If 'None', analysis
                will be done from earliest available year. Default is 'None'. 
            
            yearend (int): an intiger indicating which year analysis should be ended. If 'None', analysis
                will be done to latest available year. Default is 'None'.
        
        OUTPUT: 
            class object 
        """
    
    #figure out rest of dicts later
        
        #if data need to be preprocessed, it happens here 
        if self.preprocess==True: 
    
            # a dictionary to match the RGI numbers to the political regions (used by the WGMS) that are contained within
            #THIS SHOULD BE UPDATED AS PER THE USER SPECIFICATIONS 
            regdict={1:['US'],2:['US','CA'],3:['CA'],4:['CA'],5:['GL'],
                     6:['IS'],7:['SJ'],8:['NO','SE'],9:['RU']}
             
            #extract all political regions which may contain glaciers in our regions of interest (to help limit analysis)                         
            pol_reg=list(set([v for v in (y for x in [regdict[x] for x in regions] for y in x)]))
            
            #load glac and geo files from wgms 
            geo_dat=pd.read_csv(glob.glob(self.wgms_fp+'/*D-CHANGE.csv')[0], encoding='latin')
            glac_dat=pd.read_csv(glob.glob(self.wgms_fp+'/*EE-MASS-BALANCE.csv')[0], encoding='latin')
            
            #extract and subset relevant geodetic and glaciological data by the relevant political regions 
            geo_dat=geo_dat.loc[geo_dat['POLITICAL_UNIT'].isin(pol_reg)]
            glac_dat=glac_dat.loc[glac_dat['POLITICAL_UNIT'].isin(pol_reg)]
            #collect only full-glacier glaciolgocial mass balance values 
            glac_dat=glac_dat[(glac_dat['UPPER_BOUND'] == 9999)&(glac_dat['LOWER_BOUND'] == 9999)]                          
           
            #identify the unique glaciers found in both datasets by the WGMS ID 
            geo_sub=geo_dat['WGMS_ID'].unique()
            glac_sub=glac_dat['WGMS_ID'].unique()      
        
            #combine the two datasets and identify unique glaciers with any type of measurement 
            id_dict=pd.Series(np.unique(np.concatenate([geo_sub,glac_sub]))).rename('WGMS_ID')
        
            #load the glacier identification and lat/long data available through the wgms 
            glac_id=pd.read_csv(glob.glob(self.wgms_fp+'/*AA-GLACIER-ID-LUT.csv')[0], encoding='latin')
            glac_loc=pd.read_csv(glob.glob(self.wgms_fp+'/*A-GLACIER.csv')[0], encoding='latin')
            
            #merge all available identification and lat/long data onto the wgms ids of interest 
            id_dict=pd.merge(id_dict, glac_id[['WGMS_ID','GLIMS_ID','RGI_ID']], on='WGMS_ID', how='left')
            id_dict=pd.merge(id_dict, glac_loc[['WGMS_ID','LATITUDE','LONGITUDE']], on='WGMS_ID', how='left')
            
            #merge all RGI glacier data for regions of interest into a single csv file for ease of reference 
            rgi=pd.DataFrame()
            for x in regions:
                num=rgi_num(x)
                rgi=rgi.append(pd.read_csv(glob.glob(self.rgi_fp+ '/00_rgi60_attribs/'+num+'*.csv')[0],encoding='latin'))
                rgi60=rgi[['RGIId','GLIMSId','CenLat','CenLon','O1Region']].reset_index(drop=True)
             
                
            #create loops for each wgms id identified and try to match to rgi id
            counts=0
            for x in range(len(id_dict)):
                #rgi v.5 ids have to be treated as if they do not have an RGI number 
                
                #therefore if the RGI_ID field is null or has an RGI v. 5 id, first try to match to a GLIMS ID 
                #glims IDSc can be matched to glims ID 
                if pd.isnull(id_dict['RGI_ID'][x]) or id_dict['RGI_ID'][x].split('-')[0]=='RGI50':
                    try: 
                        id_dict['RGI_ID'][x]=rgi60['RGIId'][rgi60['GLIMSId']==id_dict['GLIMS_ID'][x]].values[0]
                        counts+=1
                    #otherwise try to match by looking at the sum and difference of lat/long coordinates in attempt to match 
                    #this method is not perfect, but attempts to find glaciers that may exist within a region 
                    except:
                        try: 
                            latlon_diff=pd.Series(abs(np.diff(rgi60[['CenLat','CenLon']], axis=1)-np.diff(id_dict.iloc[x][['LATITUDE','LONGITUDE']]))[:,0]).astype(float)
                            latlon_sums=abs(np.sum(abs(rgi60[['CenLat','CenLon']]), axis=1)-np.sum(abs(id_dict.iloc[x][['LATITUDE','LONGITUDE']])))
                            latlon_analysis=latlon_diff+latlon_sums
                            diff=rgi60[latlon_analysis<=0.004]
                            if len(diff)!=0:
                                diff['val']=latlon_diff[latlon_diff<=0.004]
                                diff=diff[diff['val']==min(diff['val'])]
                                id_dict['RGI_ID'][x]=diff['RGIId'].values[0]
                                counts+=1
                            #otherwise, glacier cannot be matched 
                            else: 
                                id_dict['RGI_ID'][x]=np.nan
                        except :
                            id_dict['RGI_ID'][x]=np.nan
            #drop glaciers that cannot be matched, and then add both a region and glacier number 
            id_dict=id_dict.dropna(subset=['RGI_ID'])
            id_dict['REGION_NO']=id_dict['RGI_ID'].apply(lambda x: int(x.split('-')[1].split('.')[0]))
            id_dict['GLAC_NO']=id_dict['RGI_ID'].apply(lambda x: int(x.split('-')[1].split('.')[1]))
            
        
        
            id_dict=id_dict[id_dict['REGION_NO'].apply(lambda x: x in regions)]
            
            temp2=[]
            # for each region, constrain by the min and max lat/long of RGI-identified glaciers for that region
            for x in regions: 
                temp=rgi60[['CenLat','CenLon']][rgi60['O1Region']==x]
                temp3=id_dict[id_dict['REGION_NO']==x]
                temp2.append(temp3[(temp3['LATITUDE']>min(temp['CenLat']))&(temp3['LATITUDE']<max(temp['CenLat']))
                          &(temp3['LONGITUDE']>min(temp['CenLon']))&(temp3['LONGITUDE']<max(temp['CenLon']))])
            id_dict=pd.DataFrame(np.concatenate(temp2), columns=id_dict.columns)
            
            #create a geo change and glac change dataset for analysis of annual mass balance 
            geo_change=pd.merge(geo_dat[['WGMS_ID','THICKNESS_CHG','THICKNESS_CHG_UNC','YEAR','SURVEY_DATE','REFERENCE_DATE']],id_dict[['WGMS_ID','RGI_ID','REGION_NO','GLAC_NO']], on='WGMS_ID',how='inner')
            glac_change=pd.merge(glac_dat[['WGMS_ID','YEAR','SUMMER_BALANCE','SUMMER_BALANCE_UNC','WINTER_BALANCE', 'WINTER_BALANCE_UNC',
                                           'ANNUAL_BALANCE','ANNUAL_BALANCE_UNC','LOWER_BOUND','UPPER_BOUND']],
                                 id_dict[['WGMS_ID','RGI_ID','REGION_NO','GLAC_NO']], on='WGMS_ID',how='inner')
            
            #change year in geo dataset to match water year rather than annual year 
            surv_month=geo_change['SURVEY_DATE'].dropna().apply(lambda x: int(str(int(x))[4:6])) 
            surv_month=surv_month[surv_month <=12][surv_month >=10].index
            
            geo_change['YEAR'][surv_month]+=1
            
            geo_change['REF_YEAR']=geo_change['REFERENCE_DATE'].dropna().apply(lambda x: int(str(int(x))[:4])) 
        
            surv_month=geo_change['REFERENCE_DATE'].dropna().apply(lambda x: int(str(int(x))[4:6])) 
            surv_month=surv_month[surv_month <=12][surv_month >=10].index
            geo_change['REF_YEAR'][surv_month]+=1
        
            #extract individual glacier area to for calculation of area-weighted mass balance 
            rgi_stat=pd.read_csv(self.rgi_fp+'/00_rgi60_summary.csv', skiprows=[0]).iloc[:21,:]
            
            rgi_stat.columns=rgi_stat.columns.map(lambda x: x.strip())
            rgi_stat=rgi_stat.dropna(subset=['O1'])
            rgi_stat[['O1','Count','Area']]=rgi_stat[['O1','Count','Area']].astype(int)
            rgi_stat.rename(columns={'Area':'Total_Area','O1':'O1Region'}, inplace=True)
            
            rgi=pd.merge(rgi, rgi_stat[['O1Region','Count','Total_Area']], on='O1Region', how='left')
        
            #convert thickness change into m w.e. 
            geo_change['THICKNESS_CHG']=(geo_change['THICKNESS_CHG']/1000)*0.9
            
            geo_change=pd.merge(geo_change,rgi[['RGIId','Area']], left_on='RGI_ID',right_on='RGIId',how='left')
            
            geo_change=geo_change.dropna(subset=['REF_YEAR'])
            
            #create the geo change mass balacne dataset 
            geo_change_mass=geo_change[['RGI_ID','REF_YEAR','YEAR','THICKNESS_CHG','Area']].rename(columns={'THICKNESS_CHG':'MASSBAL'}).dropna(subset=['MASSBAL'])
  
            #convert glaciological mass balance into m w.e. 
            glac_change['ANNUAL_BALANCE']=glac_change['ANNUAL_BALANCE']/1000
            
            #create glac change mass balance dataset 
            glac_change=glac_change.dropna(subset=['ANNUAL_BALANCE'])
            glac_change=pd.merge(glac_change,rgi[['RGIId','Area']], left_on='RGI_ID',right_on='RGIId',how='left')
            
            review_RGI=pd.DataFrame(pd.concat([geo_change_mass['RGI_ID'],glac_change['RGI_ID']]).unique()).rename(columns={0:'RGI_ID'})
            
            review_RGI['REGION_NO']=review_RGI['RGI_ID'].apply(lambda x: int(x.split('-')[1].split('.')[0]))  
        
            
            #now will convert all data into regional annual mass balacne for all years of interest 
        
            #will make 3 seperate datasets: for geodetic, glaciological, and combined mass balance 
            geo_final=[]
            glac_final=[]
            full_final=[]  
            
            #will loop by regions 
            for reg in regions: 
            
                #create region-specific lists that will hold all measurements for geo, glac, and all measurements per region \
                #easier to do per region because regional area needs to be taken into account 
                geo_array=[]
                glac_array=[]
                full_array=[]                      
            
            
                #identify all glaciers with measurements for the region currently being looped 
                REG_RGI=review_RGI[review_RGI['REGION_NO']==reg]
            
                #Select specific RGI ID 
                for x in REG_RGI['RGI_ID']: 
                    print(x)
                    #create RGI ID specific-lists that will hold all measurements available for RGI ID 
                    temp_array=[]
                    geo_temp_array=[]
                    #check if the RGI ID selected is present in the geo dataset 
                    if x in geo_change_mass['RGI_ID'].values: 
                        temp_full=geo_change_mass[geo_change_mass['RGI_ID']==x]
                        #if it is present, need to generate an array that divides geo mass bal per year
                        #will loop through every unique geodetic measurement 
                        for y in range(len(temp_full)): 
                            temp_y=temp_full.iloc[y]
                            #make a list spanned by the measurement 
                            ranges=list(range(int(temp_y['REF_YEAR']),temp_y['YEAR']))
                            #make an array of the years, and the geo measurement divided by number of years 
                            temp=np.array([ranges, np.full(len(ranges), temp_y['MASSBAL']/len(ranges)),
                                           np.full(len(ranges), temp_y['Area'])]).transpose()
                            #append this for both the geo array and array for all data 
                            temp_array.append(temp)
                            geo_temp_array.append(temp)
                        
                        #we can append the geo region-array with all the processed measurements for the unique RGI ID 
                        #for the full list (temp_array), we will need to wait to see if the RGI ID has glaciological data
                        geo_array.append(np.array(pd.DataFrame(np.concatenate(geo_temp_array,axis=0))
                                                  .groupby(by=0, as_index=False).mean()))
            
                    #see if selected RGI ID has glaciological data 
                    if x in glac_change['RGI_ID'].values: 
                        #if so, just select the columns of interest and append to both full and glac arrays for this RGI number 
                        #glac data structure does not require more work than that 
                        temp=np.array(glac_change[glac_change['RGI_ID']==x][['YEAR','ANNUAL_BALANCE','Area']])
                        glac_array.append(temp)
                        temp_array.append(temp)
            
                    #for each glacier, can only have one measurement per year
                    #if multiple measurements per year for each glacier (i.e. one from glac, one from geo), need to average
                    test2=pd.DataFrame(np.concatenate(temp_array,axis=0)).groupby(by=0, as_index=False).mean()
                    #append to region-specific list 
                    full_array.append(np.array(test2))
            
            
                #for full_array, geo_array, and glac_array, the processing steps are documented below: 
                
                #convert list of lists into dataframe 
                full_array=pd.DataFrame(np.concatenate(full_array,axis=0))
                #calculate the total area that has obs data for the year 
                full_array_sum=full_array.groupby(by=0,as_index=False).sum()[[0,2]]
                #add this total area to data 
                full_array=pd.merge(full_array,full_array_sum, on=0,how='left')
                #create column names for ease of reference 
                full_array.columns=['Year','MassBal','Glac_Area','Total_Area']
                #calculate the area weight and determine area weighted mass balance (as %of total area)
                full_array['area_weight']=full_array['Glac_Area']/full_array['Total_Area']
                full_array['MassBal_AW']=full_array['MassBal']*full_array['area_weight']
                
                #since we are interested in regional annual massbal, we group by year, and sum the area-weighted massbak 
                full_array=full_array.groupby(by='Year',as_index=False).sum()[['Year','Glac_Area','area_weight','MassBal_AW']]
                #add region number for completeness 
                full_array['REGION_NO']=[reg]*len(full_array)
                #append to the final array 
                full_final.append(full_array)
            
                #this process repeats for geodetic and glaciological only datasets if present for the specific region 
                if len(geo_array) >0:
            
                    geo_array=pd.DataFrame(np.concatenate(geo_array,axis=0))
                    geo_array_sum=geo_array.groupby(by=0,as_index=False).sum()[[0,2]]
                    geo_array=pd.merge(geo_array,geo_array_sum, on=0,how='left')
                    geo_array.columns=['Year','MassBal','Glac_Area','Total_Area']
                    geo_array['area_weight']=geo_array['Glac_Area']/geo_array['Total_Area']
                    geo_array['MassBal_AW']=geo_array['MassBal']*geo_array['area_weight']
            
                    geo_array=geo_array.groupby(by='Year',as_index=False).sum()[['Year','Glac_Area','area_weight','MassBal_AW']]
                    geo_array['REGION_NO']=[reg]*len(geo_array)
                    geo_final.append(geo_array)
            
                if len(glac_array) >0: 
                    glac_array=pd.DataFrame(np.concatenate(glac_array,axis=0))
                    glac_array_sum=glac_array.groupby(by=0,as_index=False).sum()[[0,2]]
                    glac_array=pd.merge(glac_array,glac_array_sum, on=0,how='left')
                    glac_array.columns=['Year','MassBal','Glac_Area','Total_Area']
                    glac_array['area_weight']=glac_array['Glac_Area']/glac_array['Total_Area']
                    glac_array['MassBal_AW']=glac_array['MassBal']*glac_array['area_weight']
            
                    glac_array=glac_array.groupby(by='Year',as_index=False).sum()[['Year','Glac_Area','area_weight','MassBal_AW']]
                    glac_array['REGION_NO']=[reg]*len(glac_array)
                    glac_final.append(glac_array)
            
            
            
            #put into final dataframe form 
            full_final=pd.DataFrame(np.concatenate(full_final, axis=0))
            geo_final=pd.DataFrame(np.concatenate(geo_final, axis=0))
            glac_final=pd.DataFrame(np.concatenate(glac_final, axis=0))
            
            #given column ndames
            full_final.columns=['Year','Area','AW_SUM','Spec_Massbal','REGION_NO']
            geo_final.columns=['Year','Area','AW_SUM','Spec_Massbal','REGION_NO']
            glac_final.columns=['Year','Area','AW_SUM','Spec_Massbal','REGION_NO']
            
            #adjusted to include only data from 1960 onwards for ease of visualization 
            if yearstart is not None: 
                full_final=full_final[full_final['Year']>=yearstart]
                geo_final=geo_final[geo_final['Year']>=yearstart]
                glac_final=glac_final[glac_final['Year']>=yearstart]
            if yearend is not None: 
                full_final=full_final[full_final['Year']<=yearend]
                geo_final=geo_final[geo_final['Year']<=yearend]
                glac_final=glac_final[glac_final['Year']<=yearend]  
        
            measured_id=pd.merge(id_dict,rgi[['RGIId','Area']], left_on='RGI_ID',right_on='RGIId',how='left')[['RGI_ID','REGION_NO','LATITUDE','LONGITUDE','Area']]
        #if data are already preprocessed, will simply assign the data to the appropriate variables 
        else: 
            full_final=self.wgms_fp
            geo_final=self.rgi_fp
            glac_final=self.glac_fin
            measured_id=[]
          
        #add values to object 
        self.full_final=full_final
        self.geo_final=geo_final
        self.glac_final=glac_final
        self.measured_id=measured_id
    
    def preprocess_results(self):
        """
        preprocess_results generates dataframes of mean annual mass balance with all data, only geodetic data,
        and only glaciological data, as well as a list of rgi_id and basic info of glaciers with measurement 
        
        OUTPUT:
        
        4 dataframes 
        
        """
        
        return self.full_final, self.geo_final, self.glac_final , self.measured_id

    def plot_massbal(self, regions=None, figshape=None, fig_size=(12,10),sum_stats=True, yearlims=None, 
                     area_perclim=[0,70],skip_plot=True, save_fig=False, savefp=None, figname=None):
        
        """
        plot_massbal plots the total, geodetic, and glaciological mass balance for the regions of interest 
        
        INPUT: 
            
        regions (list of n int): list of RGI regions to plot. If none, will plot all regions in self.full_final. 
            Default is 'None'. 
            
        figshape (list 2 int): the number of (rows, columns) that the plots should be arranged into. Should correspond with 
            the number of regions that are to be plotted. If 'None', fig shape is (n/2,2). Default is 'None'. 
        
        fig_size (tuple 2 float): a tuple containing the (width, height) for fig size. Default is (12,10). 
        
        sum_stats (boolean): indicate if summary statistics should be printed on the plot. Default is 'True'. 
        
        yearlims (list 2 int): a list with the start and end year that should be visualized on the plots. If 
            'None', then all years available in the datasets will be visualized. Default is 'None'. 
            
        area_perclim (list 2 int): a list of the  percent area coverage y-lim values. Default is [0, 70]. 
        
        skip_plot (boolean): indicate whether to skip the first plot and delete the extra plots when there are an odd 
            number of plots. This helps clean up figure visualization. Default is 'True'. 
            
        save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 
    
        savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
            current file location. Default is None. 
        
        filename (str): what the saved figure should be named. Default is 'Observational_Massbal'
        
        OUTPUT: 
            
            plot figure 
        
        """
        
        #determine which regions to plot 
        if regions is None: 
            regions=list(self.full_final['REGION_NO'].unique())
        
        #determine the figure shape 
        if figshape is None: 
            a=int(np.ceil(len(regions)/2))
            b=2
        else: 
            a=figshape[0]
            b=figshape[1]
        #create figure for subplots 
        fig, ax=plt.subplots(a,b, sharex='all', sharey='all', figsize=fig_size)
        fig.subplots_adjust(hspace=0.2,wspace=0)
        ax = ax.ravel()
        
        #isolate geo, glac, and full mass bal for each region 
        for counts, x in enumerate(regions):
            full_temp=self.full_final[self.full_final['REGION_NO']==x]
            geo_temp=self.geo_final[self.geo_final['REGION_NO']==x]
            glac_temp=self.glac_final[self.glac_final['REGION_NO']==x]
            
            #skip plot 1 for better visualization 
            if (skip_plot==True)&(len(regions)%b!=0):
                if counts>=b-1:
                    counts+=1
            
            #plot full massbal
            ax[counts].plot(full_temp['Year'],full_temp['Spec_Massbal'],c=colorleg['massbal'], lw=1.5)
            #create a secondary y axis to plot total area coverage 
            ax2=ax[counts].twinx()
            ax2.set_ylim(area_perclim)
            #plot geo mass bal if it exists in region
            if len(geo_temp) > 0: 
                ax[counts].plot(geo_temp['Year'],geo_temp['Spec_Massbal'],c=colorleg['geo'],ls='--', lw=0.7)
                if sum_stats==True:
                    ax[counts].text(0.025,.15,'geo:' + str(round(np.mean(geo_temp['Spec_Massbal']),4)) +' m w.e.',
                        horizontalalignment='left',transform=ax[counts].transAxes, c=colorleg['geo'])
            #plot glac mass bal if it exists in the region 
            if len(glac_temp) > 0: 
                ax[counts].plot(glac_temp['Year'],glac_temp['Spec_Massbal'],c=colorleg['glac'], ls='--', lw=0.7)
                if sum_stats==True: 
                    ax[counts].text(0.025,.25,'glac:' + str(round(np.mean(glac_temp['Spec_Massbal']),4)) +' m w.e.',
                        horizontalalignment='left',transform=ax[counts].transAxes, c=colorleg['glac'])
            #set years to be consistent for all regions 
            if yearlims is None: 
                yearlims=[min(full_temp['Year']),max(full_temp['Year'])]
            ax[counts].set_xlim(yearlims)
            #remove secondary y-axis ticks where they overlap with plot 
            if (counts+1) % b !=0:
                ax2.get_yaxis().set_visible(False)
            ax[counts].set_title(region_dict[x][0])
            ax[counts].text(0.025,.05,'total:' + str(round(np.mean(full_temp['Spec_Massbal']),4)) +' m w.e.',
                horizontalalignment='left',transform=ax[counts].transAxes, c=colorleg['massbal'])
            ax2.plot(full_temp['Year'],(full_temp['Area']/region_dict[x][1])*100, c=colorleg['neutral'],lw=1.5)
            ax[counts].axhline(0,ls=':',c=colorleg['neutral'])
            
            #make legend 
            if counts ==0:
                ax[counts].legend(['specific','geodetic','glaciological'], loc=4, bbox_to_anchor=(1.935, 0.13),frameon=False) 
                ax2.legend(['% area coverage'],loc=4,bbox_to_anchor=(2, 0),frameon=False)
        if (skip_plot==True)&(len(regions)%b!=0):
            fig.delaxes(ax[b-1])
        fig.text(0.04, 0.5, 'Mass Balance (m w.e.)', va='center', rotation='vertical', fontsize=15)
        fig.text(0.96, 0.5, '% Area Coverage', va='center', rotation=270, fontsize=15)

    
        #save plot 
        if save_fig==True: 
            if figname is None: 
                figname='Observational_Massbal'
            if savefp is None:
                plt.savefig('/'+figname+'.png')
            else:
                plt.savefig(savefp+'/'+figname+'.png') 


#model validation
                
#TO DO: 
# 1. Add documentation 
# 3. figure out the del plot thing, how to go about deleting more than one plot
# 4. add figsize option
# 5. have multiple legend options 
# 6. figure out best way for boxplot to have legend 

class model_validation: 
    def __init__(self, test_fp, valid_fp, regions, valid_type='Zemp', 
                 remove_jan=True,vcol_year='Year',vcol_mass='INT_mwe'):
        
        
        """
        class model_validation determines how well the model performs against a set of validation results 
        
        INPUT: 
        
        test_fp (list of strings): list of filepath strings for the test data NetCDF files 
        
        valid_fp (list of stings): list of filepath strings for the validation .csv files 
        
        regions (list n int): list of RGI region numbers which are being validated 
        
        valid_type (str): type of data that is validated. If 'Zemp', the first 27 rows of .csv file are removed 
            for proper formatting. Default is 'Zemp'. 
            
        remove_jan (boolean): option to remove glaciers from the island of Jan Meyen for RGI region 7 (Svalbard).
            Default is True. 
        
        vcol_year (str): the year column name for the validation data set. Default is 'Year'. 
        
        vcol_mass (str): the mass balance column name for the validation dataset. Default is 'INT_mwe'. 
        
        
        OUTPUT: 
            
        Initialized class object. 
        
        """
    
        self.test_fp=test_fp
        self.valid_fp=valid_fp
        self.regions=regions
        self.valid_type=valid_type
        self.remove_jan=remove_jan
        self.vcol_year=vcol_year
        self.vcol_mass=vcol_mass

        
        valid_mass=[]
            
    
        for counts, x in enumerate(self.regions):
            test=nc.Dataset(list(compress(self.test_fp, ['R'+str(x)+'_' in i for i in self.test_fp]))[0])
    
            #extract area and mass balance data 
            test_massbal=test['massbaltotal_glac_monthly'][:,:,0]
            #area is calculated at the beginning of every water year, so need to remove 1st year 
            test_area=test['area_glac_annual'][:,1:,0] 
       
            #remove Jan Meyen glacers from savalbrd  
            if x ==7 and self.remove_jan==True: 
                test_massbal=test_massbal[:1567,:]
                test_area=test_area[:1567,:]
    
            test_year=test['year'][:]
            #create annual, area weighted mass balance from test data 
            massbal_annual=(np.sum(test_massbal.reshape(-1,12),axis=1)).reshape(len(test_massbal),len(test_year))
            test_aw=np.sum(((massbal_annual*test_area)/np.sum(test_area,axis=0)),axis=0)
            
            
            #extract validation data 
            if self.valid_type=='Zemp':
                valid=pd.read_csv(list(compress(valid_fp, ['_'+str(x)+'_' in i for i in valid_fp]))[0],skiprows=27)
                #ensure no spaces in columns
                valid.columns=valid.columns.map(lambda t: t.strip())
                
            #constrain by dates of test data    
            valid=valid[(valid[self.vcol_year]>=test_year[0])&(valid[self.vcol_year]<=test_year[-1])]
               
            
            #test to see if the data are the same size/
            #if they are not, constrain by dates of validaiton data as well
            if len(valid)!= len(test_aw): 
                test_aw=test_aw[(test_year>=min(valid[self.vcol_year]))&(test_year<=max(valid[self.vcol_year]))]
                test_year=test_year[(test_year>=min(valid[self.vcol_year]))&(test_year<=max(valid[self.vcol_year]))]
            
            valid_mass.append(np.array([np.array(test_aw), np.array(valid[self.vcol_mass]), np.array(test_year), np.full(len(test_year),x)]))

        #append test and validation mass balance for each year for each region of interest 
        self.valid_mass=np.transpose(np.concatenate(valid_mass, axis=1))

                
    def full_results(self):
        
        """
        full_results outputs dataframe of combined test and validation results 
        
        OUTPUT: 
            
        Dataframe 
        """
        mass=pd.DataFrame(self.valid_mass, columns=['Test_MassBal','Val_MassBal','Year','Region_No'])
        return mass

    #will not encode a different stats test, will leave option opne for later
    def stats_test(self, test_type='STD'):
        
        """
        stats_test conducts the statistical test to determine the success of model in predicting mass balance 
        
        INPUT: 
            
        test_type (str): type of stats test used. Default (and currently encoded option) is standard deviation
        
        OUTPUT: 
        
            dataframe 
        """
        
        valid=[]
        
        #identify regions for which to do test 
        for x in self.regions: 
            test_aw=self.valid_mass[:,0][self.valid_mass[:,3]==x]
            valid_aw=self.valid_mass[:,1][self.valid_mass[:,3]==x]
    
            #test type; option for future encoding 
            if test_type=='STD':
                
                #determine mean and std for validation and test data 
                test_mean=np.mean(test_aw)
                test_std=np.std(test_aw)
                valid_mean=np.mean(valid_aw)
                valid_std=np.std(valid_aw)
    
                #do test 
                if (test_mean-test_std <= valid_mean <= test_mean+test_std) or (valid_mean-valid_std
                                                                                  <= test_mean <= valid_mean+valid_std) :
                    result='SIMILAR'
                    col='g'
                    print(region_dict[x][0]+': SIMILAR')
                
                else:
                    result='NOT SIMILAR'
                    col='r'
                    print(region_dict[x][0]+': NOT SIMILAR')
                    
                #append, encode and return results 
                valid.append([x,result,test_mean,test_std,valid_mean,valid_std,col])
                cols=['Region_No','Stat_Result','Test_Mean','Test_STD','Valid_Mean','Valid_STD','color']
    
            self.valid_stat=pd.DataFrame(valid, columns=cols)
        
        return self.valid_stat 
    
    def val_plot(self, regions=None, fig_shape=None, fig_size=(12,10),stats_print=True,ylim=[-2,2],
                 skip_plot=True, save_fig=False, savefp=None, figname=None): 
        
        """
        val_plot plots the test and validation mass balance datasets 
        
        INPUT: 
            
        regions (list of n int): list of RGI regions to plot. If 'None', will plot all regions identified 
            when compiling the test and validation data 
            
            
        figshape (list 2 int): the number of (rows, columns) that the plots should be arranged into. Should correspond with 
            the number of regions that are to be plotted. If 'None', fig shape is (n/2,2). Default is 'None'. 
        
        fig_size (tuple 2 float): a tuple containing the (width, height) for fig size. Default is (12,10). 
        
        ylim (list 2 float): the y lims for the mass balance of plots. Default is [-2,2]
        
        
        stats_print (boolean): indicate if summary statistics should be printed on the plot. Default is 'True'. 

        
        skip_plot (boolean): indicate whether to skip the first plot and delete the extra plots when there are an odd 
            number of plots. This helps clean up figure visualization. Default is 'True'. 
            
        save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 
    
        savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
            current file location. Default is None. 
        
        filename (str): what the saved figure should be named. Default is 'Validation_Plot'
        
        OUTPUT: 
            
            plot figure 
        """
        #compile regions 
        if regions is None: 
            regions=list(np.unique(self.valid_mass[:,3])) 
        
        #determine figure shape 
        if fig_shape is None: 
            a=int(np.ceil(len(regions)/2))
            b=2
        else: 
            a=fig_shape[0]
            b=fig_shape[1]
            
        #create figure for subplots 
        fig, ax=plt.subplots(a,b, sharex='all', sharey='all', figsize=fig_size)
        fig.subplots_adjust(hspace=0.2,wspace=0)
        ax = ax.ravel()
        
        #plot for each region
        for counts, x in enumerate(regions): 
            
            #extract test, valid and year data for each region 
            test_aw=self.valid_mass[:,0][self.valid_mass[:,3]==x]
            valid=self.valid_mass[:,1][self.valid_mass[:,3]==x]
            year=self.valid_mass[:,2][self.valid_mass[:,3]==x]
            
            #skip plots 
            if (len(regions)%b!=0)&(skip_plot==True): 
                
                if counts >= b-1: 
                    counts+=1
    
            #skip plot 1 for visualization purposes 

        
            ax[counts].plot(year,test_aw, c=colorleg['massbal'], label='modelled')
            ax[counts].plot(year,valid, c=colorleg['valid'],label='validation')
            ax[counts].axhline(0,ls=':',c=colorleg['neutral'], lw=0.5)
            ax[counts].set_ylim(ylim)
            ax[counts].set_title(region_dict[x][0])
            
            #print stats 
            if stats_print==True: 
                #try results from stats_test
                try: 
                    temp=self.valid_stat['Stat_Result'][self.valid_stat['Region_No']==x].values[0]
                    colorz=self.valid_stat['color'][self.valid_stat['Region_No']==x].values[0]
                except: 
                    #if failed, print empty stats and further instructions 
                    temp=''
                    colorz='k'
                    print('please conduct stats_test function to determine if null hypothesis can be rejected')
                
                
                ax[counts].text(0.98,.90, temp, horizontalalignment='right',transform=ax[counts].transAxes, c=colorz)
                ax[counts].text(0.98,.80, 'modelled:' + str(round(np.mean(test_aw),2)), horizontalalignment='right',transform=ax[counts].transAxes,
                              c=colorleg['massbal'])
                ax[counts].text(0.98,.70, 'test:' + str(round(np.mean(valid),2)), horizontalalignment='right',transform=ax[counts].transAxes,
                              c=colorleg['valid'])
        
        #delete axis if odd number and add labels 
        if (len(regions)%b!=0)&(skip_plot==True):     
            fig.delaxes(ax[b-1])
        fig.text(0.04, 0.5, 'Mass Balance (m.w.e.)', va='center', rotation='vertical', fontsize=15)
        ax[0].legend( loc=4, bbox_to_anchor=(2, 0))


        #save fig option
        if save_fig==True: 
            if figname is None: 
                figname='Validation_Plot'
            if savefp is None:
                plt.savefig(os.getcwd()+'/'+figname+'.png')
            else:
                plt.savefig(savefp+'/'+figname+'.png') 
        fig.show()
        
        
    
    def val_box(self, regions=None, stats_print=False,save_fig=False, savefp=None, figname=None):
        
        """
        val_box plots boxplots for validation results 
        
        INPUT: 
            
        regions (list of n int): list of RGI regions to plot. If 'None', will plot all regions identified 
            when compiling the test and validation data 
            
            
        stats_print (boolean): indicate if summary statistics should be printed on the plot. Default is 'True'. 

        save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 
    
        savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
            current file location. Default is None. 
        
        filename (str): what the saved figure should be named. Default is 'Validation_BoxPlot'
        
        OUTPUT: 
        
        boxplot figure
        
        """
        #compile list of regions 
        if regions is None: 
            regions=list(np.unique(self.valid_mass[:,3]))
            
        fig, ax=plt.subplots() 
        
        #plot for each region 
        for counts, x in enumerate(regions):
            #extract data for each region 
            test_aw=self.valid_mass[:,0][self.valid_mass[:,3]==x]
            valid=self.valid_mass[:,1][self.valid_mass[:,3]==x]
            #plot
            bp=ax.boxplot(test_aw, positions=[counts-0.05], notch=True)
            bp2=ax.boxplot(valid, positions=[counts+0.05], notch=True)
            cols=colorleg['massbal']
            
            #set boxplot style 
            for i in [bp,bp2]:
                for t in ['boxes','medians','whiskers','caps','fliers']:
                    plt.setp(i[t], color=cols)
                cols=colorleg['valid']
             
            #conduct statstest 
            if stats_print==True: 
                try: 
                    temp=self.valid_stat['Stat_Result'][self.valid_stat['Region_No']==x].values[0]
                    colorz=self.valid_stat['color'][self.valid_stat['Region_No']==x].values[0]
                except: 
                    temp=''
                    colorz='k'
                    print('please conduct stats_test function to determine if null hypothesis can be rejected')
                
                
                ax.text((counts+0.5)/len(regions),-0.35, temp, horizontalalignment='right',transform=ax.transAxes, c=colorz)
                ax.text((counts+0.5)/len(regions),-0.4, str(round(np.mean(test_aw),2)), horizontalalignment='right',transform=ax.transAxes,
                              c=colorleg['massbal'])
                ax.text((counts+0.5)/len(regions),-0.45,str(round(np.mean(valid),2)), horizontalalignment='right',transform=ax.transAxes,
                              c=colorleg['valid'])
                ax.text(-0.15,-0.35,'Stat Result:', horizontalalignment='right',transform=ax.transAxes,
                              c=colorleg['neutral'])
                ax.text(-0.15,-0.4, 'Test Massbal:', horizontalalignment='right',transform=ax.transAxes,
                              c=colorleg['massbal:'])
                ax.text(-0.15,-0.45,'Valid Massbal:', horizontalalignment='right',transform=ax.transAxes,
                              c=colorleg['valid'])
            
        #set labels     
        ax.set_ylabel('Mass Balance (m.w.e)')
        ax.set_xticks(list(range(len(regions))))
        ax.set_xticklabels( [region_dict[y][0] for y in regions], rotation = 45, ha="right")
           
        #option to save figure 
        if save_fig==True: 
            if figname is None: 
                figname='Validation_BoxPlot'
            if savefp is None:
                plt.savefig(os.getcwd()+'/'+figname+'.png')
            else:
                plt.savefig(savefp+'/'+figname+'.png') 
        fig.show()
        
    
    def component_plots(self, regions=None, fig_shape=None, fig_size=(12,10),ylim=[-2,2], y_2='frontalablation_glac_monthly', y2lim=[0,50],
                 skip_plot=True, save_fig=False, savefp=None, figname=None):
        
        """
        component plots plot the indivudal mass balance components (accumulation, refreeze, melt, frontal ablation) that 
        make up mass balance, and the change on mass balance of one of those components 
        
                    
        regions (list of n int): list of RGI regions to plot. If 'None', will plot all regions identified 
            when compiling the test and validation data 
            
            
        figshape (list 2 int): the number of (rows, columns) that the plots should be arranged into. Should correspond with 
            the number of regions that are to be plotted. If 'None', fig shape is (n/2,2). Default is 'None'. 
        
        fig_size (tuple 2 float): a tuple containing the (width, height) for fig size. Default is (12,10). 
        
        ylim (list 2 float): the y lims for the mass balance of plots. Default is [-2,2]
        
        
        y_2 (str): name (according to original netCDF file) of component for which % mass balance change should be displayed. 
            Default is 'frontalablation_glac_monthly'
            
        y2lim (list 2 float): y lim of secondary y axis that plots % mass bal change of component 

        
        skip_plot (boolean): indicate whether to skip the first plot and delete the extra plots when there are an odd 
            number of plots. This helps clean up figure visualization. Default is 'True'. 
            
        save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 
    
        savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
            current file location. Default is None. 
        
        filename (str): what the saved figure should be named. Default is 'Validation_Plot'
        
        OUTPUT: 
            
            plot figure 
        
        """
        
        #compile regions 
        if regions is None: 
            regions=list(np.unique(self.valid_mass[:,3])) 
        
        #determine figure shape 
        if fig_shape is None: 
            a=int(np.ceil(len(regions)/2))
            b=2
        else: 
            a=fig_shape[0]
            b=fig_shape[1]
            
        #create figure for subplots 
        fig, ax=plt.subplots(a,b, sharex='all', sharey='all', figsize=fig_size)
        fig.subplots_adjust(hspace=0.2,wspace=0)
        ax = ax.ravel()


        
        #plot for each region 
        for counts, x in enumerate(regions):
            
            #if number of plots isn't optimal for fig shape, skip plot 
            if (len(regions)%b!=0)&(skip_plot==True): 
                
                if counts >= b-1: 
                    counts+=1
                
            num='R'+str(int(x))+'_'
            #load relevant validation dataset and extract annual area and year 
            test=nc.Dataset(list(compress(self.test_fp, [num in i for i in self.test_fp]))[0])
            area=test['area_glac_annual'][:,1:,0]
            year=test['year']
            
            #in a loop, extract and plot mass bal components 
            temp2=[]
            for y in ['acc_glac_monthly', 'melt_glac_monthly',
                      'frontalablation_glac_monthly','refreeze_glac_monthly']: 
                temp=test[y][:,:,0]
                #convert from monthly to area-weighted annual 
                temp=(np.sum(temp.reshape(-1,12),axis=1)).reshape(len(temp),len(year))
                temp=np.sum(((temp*area)/np.sum(area,axis=0)),axis=0)
                temp2.append(temp)
                #convert the mass loss signals to negative 
                if y in ['melt_glac_monthly','frontalablation_glac_monthly']:
                    temp=-temp
                ax[counts].plot(year,temp,label=y.split('_')[0], c=colorleg[y.split('_')[0]])
                
                #determine the diff of frontal ablation, the component we are interested in 
                if y == y_2:
                    dxdy=np.diff(temp)
            
            temp2=np.sum(temp2,axis=0)
            #isolate the jump in that first year 
            dxdy=(dxdy/temp2[1:])*100
            
            #plot front ablation mass bal change on duel y axis 
            if y_2 is not None: 
                ax2=ax[counts].twinx()
                ax2.plot(year[1:],abs(dxdy),c=colorleg['neutral'], ls='-.', label='∆ mass from front abl.')
                ax2.set_ylim(y2lim)
                if (counts+1) % b !=0:
                    ax2.get_yaxis().set_visible(False)
            ax[counts].set_title(region_dict[x][0])
            ax[counts].axhline(0,ls=':',c=colorleg['neutral'], lw=0.5)

        
            #legend 
            if counts ==0: 
                ax[counts].legend( loc=4, bbox_to_anchor=(1.887, 0.13),frameon=False) 
                ax2.legend(loc=4,bbox_to_anchor=(2, 0),frameon=False)
        if (skip_plot==True)&(len(regions)%b!=0):
            fig.delaxes(ax[b-1])
        fig.text(0.04, 0.5, 'Mass Balance (m w.e.)', va='center', rotation='vertical', fontsize=15)
        fig.text(0.96, 0.5, '% Mass Change', va='center', rotation=270, fontsize=15)
        
        #save figure 
        if save_fig==True: 
            if figname is None: 
                figname='ValidationComp_Plot'
            if savefp is None:
                plt.savefig(os.getcwd()+'/'+figname+'.png')
            else:
                plt.savefig(savefp+'/'+figname+'.png') 
        fig.show()
        


#sensitiivty preprocessing and plots
        
# have options where to put the legend 
class massbal_sensitivity:
    def __init__(self, dat_fp, regions, temp_adj=[0,1,2,3], ppt_adj=4, no_ppt=True, year_range=[2000,2100], 
                 preprocess=True):
        """
        the class massbal_sensitivity processes, analyzes and visualizes simulated mass balance data from 
        PyGEM to determine mass balance sensitivity 
        
        INPUT: 
            
        dat_fp (list of strings): list of filepaths to all the mass balance simulation data of interest (in NetCDF Format). If
            preprocess=False (and processing has already been completed), this is the list of dataframes that describe 
            mean mass balance for each climate scenario for individual glaciers. 
        
        regions (list of n int): list of RGI region numbers of interest for mass balance sensitivity study. If
            preprocess=False (and processing has already been completed), this is the list of dataframes that describe 
            mean annual regional mass balance for each climate scenario. 
        
        temp_adj (list of float): list of temperature adjustments made in each simulation. The first should be the
            control (in which no modification is made). For caclulations to work properly, these numbers are assumed to 
            increase by 1. 
            
        ppt_adj (float): the percent precipitation adjustment per degree of temperature adjustment. Default is 4%/C˚.
        
        no_ppt (boolean): a boolean to indicate whether simulations in which temperature is adjusted but precipitation is not
            should be preprocessed as well. Default is True. 
            
        year_range (list 2 int): the [start year, end year] of the simulation. Default is [2000,2100]. 
        
        preprocess (boolean): option to indicate whether preprocessing is needed to extract data from the NetCDF files (True). 
            If 'False', the appropriate list of dataframes will be assigned through dat_fp and regions inputs. 
        
        OUTPUT: 
        
        Initialized class object
        
        
        """
        
        
        self.dat_fp=dat_fp #or full_galc if preprocess=0
        self.regions=regions #or full_annual if preprocess=0
        self.temp_adj=temp_adj
        self.ppt_adj=ppt_adj
        self.no_ppt=no_ppt
        self.year_range=year_range
        self.preprocess=preprocess
        #have 1 option to just return the results, and then several options for the different plots 
        
        #create a list of column names for the different temperature scenarios 
        self.cols=['T'+str(x) for x in self.temp_adj]
        
        #if preprocessing is needed
        if self.preprocess==True: 
            full_glac=[]
            full_annual=[]
            ranges=self.year_range[1]-self.year_range[0]
            #loop by temp simulation scenario 
            for temp_sim in self.temp_adj:
                #create simulation-specific lists to append to
                reg_annual=[]
                reg_glac=[]
                prec_sim=temp_sim*self.ppt_adj 
                #make filepath list of specific climate scenario for all regions 
                scen_fp=[x for x in self.dat_fp if 'T'+str(temp_sim)+'_P'+str(prec_sim) in x]
                print('T'+str(temp_sim)+'_P'+str(prec_sim))
                #create loops for each region in specific scenario 
                for reg in self.regions:
                    #load specific region in specific scenario and extract area, volume, and id data 
                    regz='R'+str(reg)
                    scen=[x for x in scen_fp if regz in x]
                    print(reg)
                    scen=nc.Dataset(scen[0])  
                    area=scen['area_glac_annual'][:,1:ranges+1,0]
                    volume=scen['volume_glac_annual'][:,:ranges,0]
                    glactable=scen['glacier_table'][:]
                    year=np.array(range(self.year_range[0],self.year_range[1]))
                    
                    #extract mass bal and component data 
                    temp_comp=[]
                    for t in ['acc_glac_monthly', 'melt_glac_monthly',
                      'frontalablation_glac_monthly','refreeze_glac_monthly', 'massbaltotal_glac_monthly']: 
                        temp=scen[t][:,:ranges*12,0]
                        #convert to annual and area-weigh data 
                        temp=np.sum(temp.reshape(-1,12),axis=1).reshape(len(temp),len(year))
                        temp2=np.array([np.sum(((temp*area)/np.sum(area,axis=0)),axis=0)])
                        temp_comp.append(temp2)
                    temp_comp=pd.DataFrame(np.concatenate(temp_comp)).T
                    temp_comp.columns=['accumulation', 'melt', 'frontal', 'refreeze', 'massbal']
                    
                     
                    
                    #load/add scenarios in which precipitation was not adjusted 
                    if (self.no_ppt==True) & (temp_sim > 0): 
                        scen=[x for x in dat_fp if 'T'+str(temp_sim)+'_P0' in x]
                        scen=[x for x in scen if 'R'+str(reg) in x]
                        scen=nc.Dataset(scen[0])  
                        massbal=scen['massbaltotal_glac_monthly'][:,:1200,0]
                        area=scen['area_glac_annual'][:,1:101,0]
                        massbal_annual=np.sum(massbal.reshape(-1,12),axis=1).reshape(len(massbal),len(year))
                        noprec_annual=np.sum(((massbal_annual*area)/np.sum(area,axis=0))
                                                                 ,axis=0)
                    else:
                        #for control scenario, to keep columns consistent, will fill with nan values 
                        noprec_annual=np.full(100, np.NaN)
    
                    #append to regional massbal dataset with relevant data for specific region         
                    reg_annual.append(pd.concat([pd.Series(year, name='year'),
                                          pd.Series(np.sum(area,axis=0), name='area'),
                                          pd.Series(np.sum(volume,axis=0),name='volume'),  
                                          pd.Series(np.full(100,reg),name='region'),temp_comp,
                                          pd.Series(noprec_annual,name='massbal_noprec')],axis=1))
                    
                    
                    #append to individual glacier dataset wit relevant data for specific region 
                    reg_glac.append(pd.concat([pd.Series(glactable[:,1], name='glac_no'), 
                                           pd.Series(np.full(len(glactable),reg),name='region'), 
                                           pd.DataFrame(glactable[:,2:4], columns=['long','lat']),
                                           pd.Series(np.mean(area,axis=1), name='area'),
                                           pd.Series(np.mean(temp,axis=1),name='mean_massbal')],axis=1))
                    
                    
                #append to scenario-specific data 
                full_glac.append(['T'+str(temp_sim)+'_P'+str(prec_sim), pd.concat(reg_glac)])
                full_annual.append(['T'+str(temp_sim)+'_P'+str(prec_sim),pd.concat(reg_annual)])
        
        #if preprocessing is not needed         
        else: 
            full_glac=self.dat_fp
            full_annual=self.regions 
  
        self.full_glac=full_glac
        self.full_annual=full_annual

    def massbal_outputs(self):
        """
        massbal_outputs creates outputs for the lists of dataframes (one dataframe for every temperature scenario)
        for mean annual regional mass balance and mean mass balance for every glacer
        
        OUTPUT: 
        
        2 lists of dataframes 
        """
        
        return self.full_glac, self.full_annual
    
    def sens_calc(self, regions=None, stats_test='wilcoxon'):
        
        """
        sens_calc calculates the mass balance sensitivity 
        
        INPUT: 
        
        regions (list of n int): list of RGI regions to plot. If 'None', will plot all regions identified 
            when compiling data in preprocessing stage. Default is 'None'. 
            
        stats_test (str): the statistical test used to determine if null hypothesis can be rejected between 
            subsequent simulations. Default is 'wilcoxon' (wilcoxon signed-rank test). No other option has 
            yet been encoded 
        
        OUTPUT: 
        list of mass balance sensitivity for each region 
        """
        
        if regions is None: 
            regions=list(self.full_annual[0][1]['region'].unique())
        
        sens=[]
        
        for x in regions: 
            sens_calc=[]
            for y in range(len(self.full_annual)):
                temp=self.full_annual[y][1]
                temp=temp[temp['region']==x]
        
                if y > 0: 
                    temp2=self.full_annual[y-1][1]
                    temp2=temp2[temp2['region']==x]
                    
                    stat=st.wilcoxon(np.array(temp['massbal']),np.array(temp2['massbal']))[1]
                    
                    #write notice if null hypothesis cannot be rejected 
                    if stat > 0.05: 
                        print(f'Null hypothesis cannot be rejected between {self.full_annual[y-1][0]} and {self.full_annual[y][0]} in {regions[x][0]}')
                        
                sens_calc.append(np.mean(temp['massbal']))
            sens.append([x,np.mean([(t-sens_calc[0])/(countz+1) for countz, t in enumerate(sens_calc[1:])])])
            self.sens=sens
        return self.sens
        
    
    def sens_box(self, regions=None, sens_sum=True, save_fig=False, savefp=None, figname=None):
        
        
        """
        sens_box plots boxplots for each scenario 
        
        INPUT: 
            
        regions (list of n int): list of RGI regions to plot. If 'None', will plot all regions identified 
            from preprocessed data. 
            
        sens_sum (boolean): indicate if summary statistics should be printed on the plot. Default is 'True'. 

        save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 
    
        savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
            current file location. Default is None. 
        
        filename (str): what the saved figure should be named. Default is 'Sensitivity_boxplots'
        
        OUTPUT: 
        
        boxplot figure
        
        """

        #compile regions 
        if regions is None: 
            regions=list(self.full_annual[0][1]['region'].unique())
            
        fig, ax=plt.subplots()
            
        #plot for each region and each temp scenario 
        for counts, x in enumerate(regions):
            position=counts*4
            for y in range(len(self.full_annual)):
                #subset data by region 
                temp=self.full_annual[y][1]
                temp=temp[temp['region']==x]
                #create boxplots for scnearios 
                bp=ax.boxplot(temp['massbal'], positions=[position], widths=0.6,
                     notch=True,patch_artist=True)
                for t in ['boxes','medians','whiskers','caps','fliers']:
                    plt.setp(bp[t], color=colorleg[self.cols[y]])
                #shift boxplot position to for next climate scenario in region 
                position+=0.5
                
            #plot stats test 
            if sens_sum==True: 
                try: 
                    ax.text(counts*4,-8.55,str(round(self.sens[counts][1],2)))
                except: 
                    print('PLEASE RUN sens_calc to calculate statistics first')
        #plot labels         
        ax.set_xticks([t*4+1 for t in list(range(len(regions)))])
        ax.set_xticklabels([region_dict[y][0] for y in regions], rotation = 45, ha="right")
        ax.set_ylabel('Mass Balance (m w.e. a$^{-1}$)')
        ax.text(-0.5,-8.55, 'Massbal Sensitivity \n (m w.e. a$^{-1}$ ˚C$^{-1}$):', horizontalalignment='right')
        pos=-4
        for t in self.cols:
            ax.text(0.3,pos,t,c=colorleg[t])
            pos-=0.3
        
        #save fig 
        if save_fig==True: 
            if figname is None: 
                figname='Sensitivity_boxplots'
            if savefp is None:
                plt.savefig(os.getcwd()+'/'+figname+'.png')
            else:
                plt.savefig(savefp+'/'+figname+'.png')   
        
        plt.show()

        
    def prec_impact(self,regions=None,stat_test='wilcoxon', save_fig=False, savefp=None, figname=None): 
                
        """
        prec_impact makes barplot to measure the relative impact of precipitation adjustment in each region
        
        INPUT: 
            
        regions (list of n int): list of RGI regions to plot. If 'None', will plot all regions identified 
            from preprocessed data. 
            
        stats_test (str): test to determine if null hypothesis can be rejected between datasets with and without
            a precipitation adjustment. Default is 'wilcoxon' (wilcoxon signed rank test). No other stats method
            has yet to be encoded. 

        save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 
    
        savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
            current file location. Default is None. 
        
        filename (str): what the saved figure should be named. Default is 'prec_impact'
        
        OUTPUT: 
        
        barplot figure
        
        """
        
        
        
        
        
        if regions is None: 
            regions=list(self.full_annual[0][1]['region'].unique())
        
        fig, ax=plt.subplots()
        #loop for relevant regions 
        for counts, x in enumerate(regions): 
            meanz=[]
            #loop for relevant climate scenarios
            for y in range(1,len(self.full_annual)): 
                #calculate mean relative % diff
                temp=self.full_annual[y][1]
                temp=temp[temp['region']==x]
                meanz.append(abs(np.mean(temp['massbal'])-np.mean(temp['massbal_noprec']))/
                             ((np.mean(temp['massbal'])+np.mean(temp['massbal_noprec']))/2))
                
                #conduct stats test 
                stat=st.wilcoxon(np.array(temp['massbal']),np.array(temp['massbal_noprec']))[1]
                    
                if stat > 0.05: 
                    print(f'Null hypothesis cannot be rejected {self.full_annual[y][0]} precitionation/no precipitation simulations in {regions[x][0]}')
            
            #plot barplots as %
            ax.bar(counts,np.mean(meanz)*100, yerr=np.std(meanz)*100, color=colorleg['massbal'])
        ax.set_xticks(list(range(len(regions))))
        ax.set_xticklabels([region_dict[k][0] for k in regions],rotation = 45, ha="right")
        ax.set_ylabel('Mass Balance (m w.e. a$^{-1}$)') 
        
        
        if save_fig==True: 
            if figname is None: 
                figname='prpec_impact'
            if savefp is None:
                plt.savefig(os.getcwd()+'/'+figname+'.png')
            else:
                plt.savefig(savefp+'/'+figname+'.png')   
        
        
        fig.show()
    
    def volume_plots(self,regions=None, figshape=None,fig_size=(15,15), sum_stats=True, skip_plot=1,
                     save_fig=False, savefp=None, figname=None):
        
        """
        volume_plot plots the volume change across temperature scenarios for each region 
        
        INPUT: 
            
        regions (list of n int): list of RGI regions to plot. If 'None', will plot all regions identified 
           during data preprocessing
            
            
        figshape (list 2 int): the number of (rows, columns) that the plots should be arranged into. Should correspond with 
            the number of regions that are to be plotted. If 'None', fig shape is (n/2,2). Default is 'None'. 
        
        fig_size (tuple 2 float): a tuple containing the (width, height) for fig size. Default is (12,10). 
        
        
        sum_stats (boolean): indicate if summary statistics should be printed on the plot. Default is 'True'. 

        
        skip_plot (boolean): indicate whether to skip the first plot and delete the extra plots when there are an odd 
            number of plots. This helps clean up figure visualization. Default is 'True'. 
            
        save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 
    
        savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
            current file location. Default is None. 
        
        filename (str): what the saved figure should be named. Default is 'Volume_Plots'
        
        OUTPUT: 
            
        plot figure 
        """
       
        if regions is None: 
            regions=list(self.full_annual[0][1]['region'].unique())
        
        if figshape is None: 
            a=int(np.ceil(len(regions)/2))
            b=2
        else: 
            a=figshape[0]
            b=figshape[1]
        
        fig,ax=plt.subplots(a,b, sharex='all', sharey='all' , figsize=fig_size)
        fig.subplots_adjust(hspace=0.2,wspace=0)
        ax = ax.ravel()
        
        #loop for regions 
        for counts, x in enumerate(regions): 
            
            if (skip_plot==1)&(len(regions)%b!=0):
                if counts>=b-1:
                    counts+=1
            
            #plot volume as % of initial volume 
            for y in range(len(self.full_annual)): 
                temp=self.full_annual[y][1]
                temp=temp[temp['region']==x]
                ax[counts].plot(temp['year'], (temp['volume']/temp['volume'][0])*100, label=self.cols[y], c=colorleg[self.cols[y]])
                
            ax[counts].set_title(region_dict[x][0])
            if sum_stats==True:
                ax[counts].text(0.05,0.25,'Initial Volume:'+ str(round(temp['volume'][0]))+'km3',transform=ax[counts].transAxes)
                ax[counts].text(0.05,0.15, 'T'+self.cols[-1]+'Volume:'+ str(round(list(temp['volume'])[-1]))+'km3',transform=ax[counts].transAxes)
        
        if (skip_plot==1)&(len(regions)%b!=0):
            fig.delaxes(ax[b-1])
            
        fig.text(0.04, 0.5, '% Volume', va='center', rotation='vertical', fontsize=15)
        ax[0].legend( loc=4, bbox_to_anchor=(2, 0))    
        
        if save_fig==True: 
            if figname is None: 
                figname='Volume_Plots'
            if savefp is None:
                plt.savefig(os.getcwd()+'/'+figname+'.png')
            else:
                plt.savefig(savefp+'/'+figname+'.png')  
  
        fig.show()
        
    def component_plots(self,regions=None, stats_plot=True, save_fig=False, savefp=None, figname=None):
        
        """
        component_plots creates stacked barplot of normalized mass balance components (melt, accumulation, 
        frontal ablation, and refreeze) for each region and each scenario 
        
        INPUT: 
            
        regions (list of n int): list of RGI regions to plot. If 'None', will plot all regions identified 
           during data preprocessing
            
        
        stats_plot (boolean): indicate if summary statistics should be printed on the plot. Default is 'True'. 

        
        skip_plot (boolean): indicate whether to skip the first plot and delete the extra plots when there are an odd 
            number of plots. This helps clean up figure visualization. Default is 'True'. 
            
        save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 
    
        savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
            current file location. Default is None. 
        
        filename (str): what the saved figure should be named. Default is 'Component_Plots'
        
        OUTPUT: 
            
        stacked barplot figure
        """

        if regions is None: 
            regions=list(self.full_annual[0][1]['region'].unique())
        
        fig, ax = plt.subplots(len(regions),1, sharex='all', figsize=(7,14))
        fig.subplots_adjust(hspace=.2)
        ax = ax.ravel()
        #loop for relevant areas 
        for counts, x in enumerate(regions):
        
            temps=[]
            #loop for scenario 
            for y in range(len(self.full_annual)):
                temp=self.full_annual[y][1]
                temp=temp[temp['region']==x]
                
                #extract compponents 
                temp=np.mean(temp[['accumulation', 'melt', 'frontal', 'refreeze']], axis=0)
                #append as normalized %
                temps.append((temp/sum(temp))*100)
            #expand out components for stacked barplot    
            melt=[temps[b][1] for b in [0,1,2,3]]
            front=[temps[b][2] for b in [0,1,2,3]]
            acc=[temps[b][0] for b in [0,1,2,3]]
            ref=[temps[b][3] for b in [0,1,2,3]]
            
            #create stacked barplot 
            ax[counts].bar([0,1,2,3],melt,color=colorleg['melt'])
            ax[counts].bar([0,1,2,3],front, color=colorleg['frontalablation'], bottom=melt)
            ax[counts].bar([0,1,2,3],acc, color=colorleg['acc'],bottom=[i+j for i,j in zip(melt,front)])
            ax[counts].bar([0,1,2,3],ref, color=colorleg['refreeze'],bottom=[i+j+k for i,j,k in zip(melt, front,acc)])
            ax[counts].tick_params(axis='y',labelsize=16, top=False)
            ax[counts].get_yticklabels([-1])
            ax[counts].set_ylim([0,100])
            ax[counts].set_title(region_dict[x][0])
        
            #calculate massbal sensitivity of individual components 
            melt=np.mean([(t-melt[0])/(countz+1) for countz, t in enumerate(melt[1:])])
            front=np.mean([(t-front[0])/(countz+1) for countz, t in enumerate(front[1:])])
            acc=np.mean([(t-acc[0])/(countz+1) for countz, t in enumerate(acc[1:])])
            ref=np.mean([(t-ref[0])/(countz+1) for countz, t in enumerate(ref[1:])])
            
            ax[counts].text(1.03,.9, str(round(ref,2)), horizontalalignment='left',transform=ax[counts].transAxes,
                          c=colorleg['refreeze'], fontsize=12)
            ax[counts].text(1.03,.75, str(round(acc,2)), horizontalalignment='left',transform=ax[counts].transAxes,
                          c=colorleg['acc'], fontsize=12)
            ax[counts].text(1.03,.6, str(round(front,2)), horizontalalignment='left',transform=ax[counts].transAxes,
                          c=colorleg['frontalablation'], fontsize=12)
            ax[counts].text(1.03,.45, str(round(melt,2)), horizontalalignment='left',transform=ax[counts].transAxes,
                          c=colorleg['melt'], fontsize=12)
        
        #plt.xlabel('Temperature Increase')
        if stats_plot==True: 
            ax[0].text(1.06,1.2, 'massbal component \n sensitivity (% a$^{-1}$ ˚C$^{-1}$)', horizontalalignment='center',transform=ax[0].transAxes,
                               fontsize=12)
            ax[0].legend(['Melt','Frontal Ablation','Accumulation','Refreeze'],loc=1, bbox_to_anchor=(1.6, 1))
        plt.xticks(list(range(len(self.cols))),['T'+str(t) for t in self.cols], fontsize=16)
        
        
        if save_fig==True: 
            if figname is None: 
                figname='Component_Plots'
            if savefp is None:
                plt.savefig(os.getcwd()+'/'+figname+'.png')
            else:
                plt.savefig(savefp+'/'+figname+'.png') 
        
        plt.show()
        

#hist plots
#documentation/debug/verbose
def hist_plots(dat_fp, regions,feature=['Area','Zmed','Slope','TermType'],
               figshape=None,fig_size=(12,10), stats_print=True, skip_plot=True, remove_jan=True,
               log_plot=None, save_fig=False, savefp=None, figname=None):
    
    """
    hist_plot creates histograms for the physical aspects of glaciers as per the RGI
    
    INPUT: 
        
    regions (list of n int): list of RGI regions to plot. 
        
    feature (list of str): features, as identified by the RGI to plot. Default is: ['Area','Zmed','Slope','TermType']. 
        
    figshape (list 2 int): the number of (rows, columns) that the plots should be arranged into. Should correspond with 
        the number of regions that are to be plotted. If 'None', fig shape is (n/2,2). Default is 'None'. 
    
    fig_size (tuple 2 float): a tuple containing the (width, height) for fig size. Default is (12,10). 
    
    
    stats_print (boolean): indicate if summary statistics should be printed on the plot. Default is 'True'. 

    
    skip_plot (boolean): indicate whether to skip the first plot and delete the extra plots when there are an odd 
        number of plots. This helps clean up figure visualization. Default is 'True'. 
        
    remove_jan (boolean): indicate whether the island of Jan Mayen should be removed from analysis of Svalbard (R7).
        Default is 'True'. 
        
    log_plot (int): indiacte whether the plot should have a logarithmic scale. 1 = all plots should have algorithmic scale, 
        2=none of the plots should have logarithmic scale. None=pre-defined plots will have logarithmic scale while 
        others will not. Default is 'None'. 
        
    save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 

    savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
        current file location. Default is None. 
    
    filename (str): what the saved figure should be named. Default is '[name of feature]_plots'
    
    OUTPUT: 
        
        histogram figure 
    """    
  
    if figshape is None: 
        a=int(np.ceil(len(regions)/2))
        b=2
    else: 
        a=figshape[0]
        b=figshape[1]
    
    
    for feat in feature: 

        if (log_plot==1) | ((feat in ['Area','TermType']) & (log_plot==None)):
            logs=True
            fig, ax=plt.subplots(a,b,sharex='all', figsize=fig_size)
        elif ((feat not in ['Area','TermType']) & (log_plot==None))| (log_plot==2):
            logs=False
            fig, ax=plt.subplots(a,b,sharex='all', sharey='all', figsize=fig_size)
        fig.subplots_adjust(hspace=0.3,wspace=0.2)
        ax = ax.ravel()
        
        
        #now for each feature, need to populate subplots for each region 
        for counts, x in enumerate(regions): 
            
            #read relevant cvs file from filepath list 
            temp=pd.read_csv([y for y in dat_fp if '0'+str(x) + '_rgi60' in y][0], encoding='latin')
    
    
            #because we are not interested in the island of Jan Mayen, the glaciers located there need to be removed from R7. 
            if remove_jan==True:
                if x == 7: 
                    temp=temp.iloc[:1567,:]
            
            #because we have an odd number of plots, want to leave plot 1 empty for better visualization 
            #basically, after plot 0 is populated, skip plot 1 
            if (skip_plot==1)&(len(regions)%b!=0):
                if counts>=b-1:
                    counts+=1
                
                
            ax[counts].hist(temp[feat],density=True, log=logs, color=colorleg['glacier'])
            ax[counts].set_title(region_dict[x][0], fontdict={'fontsize':15})
            ax[counts].tick_params(axis='both', which='major', labelsize=15)
            if stats_print==True:
                # unless feature is termtype, want to print some mean/med
                if feat!='TermType':
                    ax[counts].text(0.98,0.9,'median :'+str(round(np.median(temp[feat]),3)), transform=ax[counts].transAxes,
                                 horizontalalignment='right')
                    ax[counts].text(0.98,0.75,'mean :'+str(round(np.mean(temp[feat]),3)), transform=ax[counts].transAxes,
                                 horizontalalignment='right')  
                else: 
                    ax[counts].text(0.98,0.9,'% land term :'+str(round((len(temp[feat][temp[feat]==0])/len(temp[feat]))*100,3)), transform=ax[counts].transAxes,
                                 horizontalalignment='right')
                    ax[counts].text(0.98,0.75,'% marine term :'+str(round((len(temp[feat][temp[feat]==1])/len(temp[feat]))*100,3)), transform=ax[counts].transAxes,
                                 horizontalalignment='right')  
        fig.suptitle(feat, fontsize=18)
        #delete plot 1, which is not populated 
        
        if (skip_plot==True)&(len(regions)%b!=0):
            fig.delaxes(ax[b-1])
            
        if save_fig==True: 
            if figname is None: 
                figname=feat+'_plots'
            if savefp is None:
                plt.savefig(os.getcwd()+'/'+figname+'.png')
            else:
                plt.savefig(savefp+'/'+figname+'.png')      
            
        plt.show()


#hyps plots 
        

def hyps_plot(dat_fp, regions, percent=False, figshape=None,fig_size=(15,10), stats_print=True, 
              skip_plot=True,  save_fig=False, savefp=None, figname=None):

    """
    hyps_plot creates plots from 10m elevation bins to look at distribution of glacier volume on glaciers 
    
    INPUT: 
        
    regions (list of n int): list of RGI regions to plot. 
    
    percent (boolean): inidcate whether to plot a barplot of % volume in elevation bins (True), or volume in km3 in
        each elevation bin (False). Default is 'False'. 
        
    figshape (list 2 int): the number of (rows, columns) that the plots should be arranged into. Should correspond with 
        the number of regions that are to be plotted. If 'None', fig shape is (n/2,2). Default is 'None'. 
    
    fig_size (tuple 2 float): a tuple containing the (width, height) for fig size. Default is (15,10). 
    
    stats_print (boolean): indicate if summary statistics should be printed on the plot. Default is 'True'. 
    
    skip_plot (boolean): indicate whether to skip the first plot and delete the extra plots when there are an odd 
        number of plots. This helps clean up figure visualization. Default is 'True'. 
        
    save_fig (boolean): indicate whether the figure produced by this function should be saved. Default is False. 

    savefp (str): the loaction to which the figure should be saved. If 'None', figure will be saved to 
        current file location. Default is None. 
    
    filename (str): what the saved figure should be named. Default is 'Cumu_Volume'or 'Percent_Volume'
    
    OUTPUT: 
        
        histogram figure 
    """    

    #determine figshape
    if figshape is None: 
        a=int(np.ceil(len(regions)/2))
        b=2
    else: 
        a=figshape[0]
        b=figshape[1] 

    fig, ax= plt.subplots(a,b,sharex='all', sharey='all', figsize=fig_size)
    fig.subplots_adjust(hspace=0.2,wspace=0) 
    ax=ax.ravel()
    
    #store max number of bins for x-axis plotting 
    maxnums=[]
    #create loop to populate plots in each region 
    for counts, x in enumerate(regions):
        
        #pull relevant files from list of filepaths 
        thickness=[y for y in dat_fp if 'thickness_RGI0'+str(x) in y]
        area=[y for y in dat_fp if 'area_RGI0'+str(x) in y]  
    
        #here we replace the -99 with 0, to indicate that these data points have no ice for numeric calculation
        area=pd.read_csv(area[0]).iloc[:,2:].replace(-99,0)
        thickness=pd.read_csv(thickness[0]).iloc[:,2:].replace(-99,0)
    
        #calculate volume from thickness and area, and extract altitude 
        volume=area.values*thickness.values
        volume=volume.sum(axis=0)
        alt=area.columns.values.astype(int)
        
    
        #because we have an odd number of plots, want to leave plot 1 empty for better visualization 
        #basically, after plot 0 is populated, skip plot 1 
        if (skip_plot==True)&(len(regions)%b!=0):
            if counts>=b-1:
                counts+=1
        if percent==False: 
            title='Cumu_Volume'
            ax[counts].plot(volume, alt, c=colorleg['glacier'])
            ax[counts].set_title(region_dict[x][0])
            ax[counts].fill_between(volume,alt, color=colorleg['glacier'])
        #print total glacier volume on plot 
            if stats_print==True: 
                ax[counts].text(0.98,0.9,'Total Glac Volume:' + str(round(sum(volume)/1000))[:-2]+ ' x10$^{3}$ km$^{3}$', transform=ax[counts].transAxes,
                         horizontalalignment='right')
        
        elif percent==True: 
            title='Percent_Volume'
            #for percentage, putting data into 500m elevation bins visualizing using barplot
            temp=np.add.reduceat(volume, np.arange(0, len(volume), 50))
            maxnums.append(len(temp))
            ax[counts].bar(range(len(temp)),(temp/sum(volume))*100, color=colorleg['glacier'], width=1, align='edge',
                            edgecolor=colorleg['neutral'])
            ax[counts].set_title(region_dict[x][0])
            ax[counts].set_xticks(list(range(0,max(maxnums),2)))
            ax[counts].set_xticklabels([t*500 for t in list(range(0,max(maxnums),2))]) 
            #adding numbers to the barplot for better interpretability 
            for countz, y in enumerate(temp):
                ax[counts].text(countz+0.5,((y/sum(volume))*100)+6, str(round((y/sum(volume))*100,2)),horizontalalignment='center',
                                c=colorleg['neutral'], fontweight='bold')
                ax[counts].text(countz+0.5,((y/sum(volume))*100)+1, str(round(y/1000))[:-2],horizontalalignment='center',
                                c=colorleg['glacier'], fontweight='bold')
    if percent==False: 
        fig.suptitle('Cumultive Glacier Volume Per Elevation Band', fontsize=25)
        fig.text(0.5, 0.04, 'Volume (km$^{3}$)', ha='center', fontsize=15)
        fig.text(0.04, 0.5, 'Altitude (m a.s.l)', va='center', rotation='vertical', fontsize=15)
     
    elif percent==True: 
        fig.suptitle('Percent Glacier Volume Per Elevation Band', fontsize=25)
        fig.text(0.5, 0.04, 'Altitude (m a.s.l)', ha='center', fontsize=15)
        fig.text(0.04, 0.5, 'Volume (%)', va='center', rotation='vertical', fontsize=15)
        ax[0].text(2,0.1,'glaier volume (%)', ha='right', transform=ax[0].transAxes, c=colorleg['neutral'],fontweight='bold')
        ax[0].text(2,0,'glaier volume (x10$^{3}$ km$^{3}$)', ha='right', transform=ax[0].transAxes,c=colorleg['glacier'],fontweight='bold')
    
    if (skip_plot==True)&(len(regions)%b!=0):
        fig.delaxes(ax[b-1])

    if save_fig==True: 
        if figname is None: 
            figname=title
        if savefp is None:
            plt.savefig(os.getcwd()+'/'+figname+'.png')
        else:
            plt.savefig(savefp+'/'+figname+'.png')      
            
        plt.show()





