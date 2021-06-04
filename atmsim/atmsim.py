#!/usr/bin/env python3
'''Test'''

import argparse
import matplotlib.pyplot as plt
import numpy as np
import toast
import toast.tod
import toast.todmap
import toast.pipeline_tools
from toast.weather import Weather
from toast.utils import Environment
from argparse import ArgumentParser
import healpy as hp
import astropy.units as u

from todgb import TODGb
from sda_mod import OpSimAtmosphereMod


# Default parameters
START_TIME = 1577839800.0
DURATION = 60*60*24

SAMPLE_RATE = 100 #Hz
SCAN_SPEED = 120 # deg/s

# observatory
OT_ALTITUDE = 2390 # m
OT_LON = '{}'.format(-(17 + 29*1/60 + 16*1/60/60))
OT_LAT = '{}'.format(28 + 18*1/60 + 8*1/60/60)

# map making
NSIDE = 32

# Obs name
OBS_NAME = 'gb_atm'

def setup(comm, args):
    '''Setup the observation'''

    # focal plane
    fp_tmp = toast.pipeline_tools.Focalplane(fname_pickle=args.fp_path)
    totsamples = int(args.duration*args.sample_rate)

    if comm.comm_group is not None:
        ndetrank = comm.comm_group.size
    else:
        ndetrank = 1

    todgb = TODGb(comm.comm_group,
                  fp_tmp.detquats,
                  totsamples,
                  firsttime=args.start_time,
                  el=args.elevation,
                  site_lon=OT_LON,
                  site_lat=OT_LAT,
                  site_alt=OT_ALTITUDE,
                  scanrate=args.scan_speed,
                  detranks=ndetrank,
                  boresight_angle=args.boresight_angle)

    obs = {}
    obs['name'] = OBS_NAME
    obs['tod'] = todgb
    obs['noise'] = fp_tmp.noise
    obs['id'] = 2725
    obs['rate'] = 100

    obs['site_name'] = 'Teide Observatory'
    obs['site_id'] = 128
    obs['altitude'] = OT_ALTITUDE
    obs['weather'] = Weather("weather_Atacama.fits")
    obs['telescope_name'] = 'GroundBIRD'
    obs['telescope_id'] = 0
    obs['focalplane'] = fp_tmp.detector_data
    obs['fpradius'] = fp_tmp.radius
    obs['start_time'] = args.start_time
    obs['season'] = 2020
    obs['date'] = '2020-0101'
    obs['MJD'] = 58849.034722
    obs['rising'] = False
    obs['mindist_sun'] = 86.36308327288089
    obs['mindist_moon'] = 40.32976225720199
    obs['el_sun'] = -18.68

    data = toast.Data(comm)
    data.obs.append(obs)

    return data


def argument_parser():
    '''Argument parser'''
    parser = ArgumentParser()

    # Observation setting
    parser.add_argument('--start_ut', default=START_TIME, type=float, help='Start time in Unix time.')
    parser.add_argument('--duration', default=DURATION, type=float, help='Duration in seconds.')
    parser.add_argument('--sample_rate', default=SAMPLE_RATE, type=float, help='Sample rate in Hz')

    # Telescope setting
    parser.add_argument('--fp_path', default='./sron.pkl', type=str, help='Path to the focalplane info file.')
    parser.add_argument('--elevation', default=60, type=float, help='Telescope elevation in degree.')
    parser.add_argument('--scan_speed', default=120, type=float, help='Scan speed in deg/s.')
    parser.add_argument('--boresight_angle', default=0, type=float, help='Boresight angle in degree.')


    # Simulation setting
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--zmax', default=1000, type=float,
                        help='Maximum altitude to integrate')
    parser.add_argument('--zatm', default=40000.0, type=float,
                        help='atmosphere extent for temperature profile.')
    parser.add_argument('--z0_center', default=2000.0, type=float,
                        help='central value of the water vapor distribution.')
    parser.add_argument('--z0_sigma', default=0.0, type=float,
                        help='sigma of the water vapor distribution.')

    parser.add_argument('--lmin_center', default=0.01, type=float,
                        help='Dissipation scale')
    parser.add_argument('--lmin_sigma', default=0, type=float)
    parser.add_argument('--lmax_center', default=300, type=float,
                        help='Injection scale')
    parser.add_argument('--lmax_sigma', default=0, type=float)

    parser.add_argument('--xstep', default=30, type=float,
                        help='size of volume elements in X direction.')
    parser.add_argument('--ystep', default=30, type=float,
                        help='size of volume elements in Y direction.')
    parser.add_argument('--zstep', default=30, type=float,
                        help='size of volume elements in Z direction.')

    parser.add_argument('--nelem', default=10000, type=int,
                        help='controls the size of the simulation slices.')
    parser.add_argument('--gain', default=3e-5, type=float,
                        help='Fluctuation gain')

    parser.add_argument('--common_flag_name', default=None,
                        help='Cache name of the output common flags.')
    parser.add_argument('--common_flag_mask', default=255,
                        help='Bitmask to use when flagging data based on the common flags.')
    parser.add_argument('--flag_name', default=None)
    parser.add_argument('--flag_mask', default=255)
    parser.add_argument('--apply_flags', default=False, type=bool)
    parser.add_argument('--report_timing', default=True, type=bool,
                        help='Print out time taken to initialize, simulate and observe')

    parser.add_argument('--wind_dist', default=10000, type=float,
                        help='Maximum wind drift before discarding the volume and creating a new one [meters].')
    parser.add_argument('--cachedir', default=None, type=str,
                        help='Directory to use for loading and saving atmosphere realizations. Set to None to disable caching.')
    parser.add_argument('--freq', default=None, type=float,
                        help='Observing frequency in GHz.')
    parser.add_argument('--write_debug', default=True, type=bool,
                        help='If True, write debugging files.')

    args = parser.parse_args()

    return args


def main():
    '''Main function
    '''

    args = argument_parser()

    env = toast.Environment.get()
    env.set_log_level('DEBUG')
    
    mpiworld, procs, rank = toast.mpi.get_world()
    comm = toast.mpi.Comm(mpiworld, procs)

    data = setup(comm, args)

    # Atmospheric simulation
    atmsim = OpSimAtmosphereMod(
        out="atm_90",
        realization=args.seed,
        component=123456,
        lmin_center=args.lmin_center,
        lmin_sigma=args.lmin_sigma,
        lmax_center=args.lmax_center,
        lmax_sigma=args.lmax_sigma,
        zatm=args.zatm,
        zmax=args.zmax,
        xstep=args.xstep,
        ystep=args.ystep,
        zstep=args.zstep,
        nelem_sim_max=args.nelem,
        gain=args.gain,
        z0_center=args.z0_center,
        z0_sigma=args.z0_sigma,
        apply_flags=args.apply_flags,
        common_flag_name=args.common_flag_name,
        common_flag_mask=args.common_flag_mask,
        flag_name=args.flag_name,
        flag_mask=args.flag_mask,
        report_timing=args.report_timing,
        wind_dist=args.wind_dist,
        cachedir=args.cachedir,
        freq=args.freq,
        plot=False,
        write_debug=args.write_debug
    )
    atmsim.exec(data)
    
    toast.todmap.OpPointingHpix(nside=NSIDE, nest=True, mode="IQU").exec(data)
    
    destriped_name = "destriped_90"
    toast.tod.OpCacheCopy(input='atm_90', output=destriped_name, force=True).exec(data)

    mapmaker = toast.todmap.OpMapMaker(
        nside=NSIDE,
        nnz=3,
        name=destriped_name,
        outdir='maps_90',
        outprefix="toast_test_",
        baseline_length=10,
        iter_max=100,
        use_noise_prior=False,
    )
    mapmaker.exec(data)


if __name__ == '__main__':
    main()
