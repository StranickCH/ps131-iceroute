@echo off
rem Full recomputation of the paper, in order. Run from the Anaconda Prompt in this folder.
rem Finished downloads, cached frames and features are skipped on a restart; routing and PPO runs
rem create a new dated folder under runs\ each time, and the latest folder of each tag is used.
rem Time on a desktop PC (CPU, one GPU for the image features): downloads several hours,
rem classification ~1 h, routing ~25 runs x 40-50 min = ~19 h, PPO 3 seeds x ~7 h = ~21 h.
rem Before the DMI step: pip install copernicusmarine and run "copernicusmarine login" once.
set PYTHONUTF8=1
cd /d "%~dp0"
set BASE=--matrix-cv 5 --obs-model resample --trials 20

rem ---- 1. data, frame index, labels (Sect. 3)
python download_ps131_full.py || goto fail
python build_ps131_index.py || goto fail
python collocate_ps131.py || goto fail
python label_ps131_synop.py || goto fail
python prepare_ps131_dataset.py --max-dt 180 --cache || goto fail

rem ---- 2. classification, calibration, checks (Sects. 4.1-4.3, 5.1-5.2)
python train_ps131_cv.py --mode probe || goto fail
python train_ps131_cv.py --mode finetune --seeds 3 || goto fail
python fusion_ps131.py || goto fail
for /f "delims=" %%d in ('dir /b /ad /o:n runs\fusion_2*') do set OOF=runs\%%d\fusion_oof.csv
echo OOF = %OOF%
python cv_repeat_ps131.py --oof %OOF% --repeats 10 || goto fail
python check_codes_temporal_ps131.py || goto fail

rem ---- 3. DMI ice charts (alternative reference, Sect. 3.3)
python download_dmi_charts_ps131.py || goto fail
python check_dmi_coverage_ps131.py --oof %OOF% || goto fail

rem ---- 4. routing (Sects. 4.4-4.5, 5.3-5.6); quick check first: add --smoke --tag smoke
python route_ps131_v3.py --oof %OOF% %BASE% --tag main32 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --max-label-dt 60 --tag lab60 || goto fail
python route_ps131_v3.py --oof %OOF% --obs-model resample --trials 20 --time-split --tag timesplit32 || goto fail
python route_ps131_v3.py --oof %OOF% --obs-model matrix --trials 20 --time-split --tag timesplit_matrix || goto fail
python route_ps131_v3.py --oof %OOF% --matrix-cv 5 --obs-model matrix --trials 20 --tag matrix || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --floe-sigma 3 --tag floe3 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --floe-sigma 12 --tag floe12 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --thr 0.25,0.75 --tag thr25_75 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --thr 0.45,0.90 --tag thr45_90 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --class-cost 1,4,8 --tag cost148 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --class-cost 1,2,12 --tag cost1212 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --class-cost 1,2,4 --tag cost124 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --cam-range-scale 2 --tag range2 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --cam-block 1 --tag block1 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --cam-block 6 --tag block6 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --cvar-alpha 0.5 --tag cvar050 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --cvar-alpha 0.9 --tag cvar090 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --cam-donor --tag donor32 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --truth dmi --dmi-min-cover 0.7 --tag dmi70 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --truth dmi --dmi-min-cover 0.9 --tag dmi90 || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --truth dmi --dmi-min-cover 0.7 --dmi-nodata water --tag dmi70water || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --truth dmi --dmi-min-cover 0.7 --dmi-nodata heavy --tag dmi70heavy || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --truth dmi --dmi-min-cover 0.7 --dmi-nodata amsr --tag dmi70amsr || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --truth dmi --dmi-min-cover 0.5 --dmi-nodata amsr --tag dmi50amsr || goto fail
python route_ps131_v3.py --oof %OOF% %BASE% --truth dmi --dmi-min-cover 0.7 --cam-donor --tag dmi70donor || goto fail

rem ---- 5. PPO agent, three training seeds (Sects. 4.6, 5.7); quick check first: add --smoke
python drl_route_ps131_v3.py --oof %OOF% --train-seed 0 --hours 6 || goto fail
python drl_route_ps131_v3.py --oof %OOF% --train-seed 1 --hours 6 || goto fail
python drl_route_ps131_v3.py --oof %OOF% --train-seed 2 --hours 6 || goto fail
python drl_seeds_summary.py || goto fail

rem ---- 6. extra analyses, figures, supplementary material, results archive
python reviewer_analyses_ps131.py . || goto fail
python make_figures_article3.py --oof %OOF% --lang en || goto fail
python make_figures_shrink.py runs runs\article3_figs_en || goto fail
python make_supplement_en.py runs runs\supplement_en || goto fail
python pack_results.py . paper_results.zip || goto fail
echo Done. Figures: runs\article3_figs_en, supplementary: runs\supplement_en
pause
exit /b 0
:fail
echo Stopped with an error at the step above.
pause
exit /b 1
