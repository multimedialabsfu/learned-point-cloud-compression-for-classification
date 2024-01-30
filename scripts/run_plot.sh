#!/bin/bash

# COMPRESSAI_PLOT=(poetry run compressai-plot)

PLOT=(
  # "${COMPRESSAI_PLOT[@]}"
  poetry run compressai-plot
  --aim_repo="$HOME/data/aim/pc-mordor/pcc"
)

COMMON_ARGS=(--x='bpp_loss' --y='acc_top1' --optimal="convex")
# COMMON_ARGS=(--x='bpp_loss' --y='acc_top1')

# Useful queries:
# run.created_at >= datetime(2023, 5, 16)
# run.dataset.train.meta.name == "ModelNet40"

# # FULL_CSV generated by:
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='"pcc-cls-only-pointnet" in run.model.name and run.hp.num_channels.g_a == [3,64,64,64,128,1024]' \
#   --out_csv=results/plot_rd/modelnet40_bpp_loss_full.csv \
#   --out_html=results/plot_rd/modelnet40_bpp_loss_full.html
#
# # LITE_CSV generated by:
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='"pcc-cls-only-pointnet" in run.model.name and run.hp.num_channels.g_a == [3,8,8,16,16,32]' \
#   --out_csv=results/plot_rd/modelnet40_bpp_loss_lite.csv \
#   --out_html=results/plot_rd/modelnet40_bpp_loss_lite.html
#
# # AGGREGATE_NEW_TSV generated using:
# poetry run python scripts/eval_modelnet.py


# # Original lite query:
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='"pcc" in run.model.name and run.hp.num_channels.g_a == [3,8,8,16,16,32]' \
#   --out_csv=results/plot_rd/modelnet40_bpp_loss_lite.csv \
#   --out_html=results/plot_rd/modelnet40_bpp_loss_lite.html



# # run.hp.num_points=1024
# # run.hp.num_channels.g_a=[3,8,64]
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='run.hash in "9e12dd9929fe4a02adfa849d d43385b8f039447d85c78171 574386ebe91a4571a88f558f b68f7630627e4238b40bcc2f 61471e16d91f4b24bd2a9c8e dc54ca5b5c0c4a52ba00da0e 023ff4ecb494450ab2c336dd b63dcdbd7ba5459995b2212d d3756ef84fa246f4846c58a0 4c1510780fe94923aaeba1ef 50e59b3e01294da391d6e212 aca683c052df45c5a44d1f48".split()' \
#   --out_csv=results/plot_rd/tmp.csv \
#   --out_html=results/plot_rd/tmp.html

# # run.hp.num_points=256
# # run.hp.num_channels.g_a=[3,8,64]
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='run.hash in "b98114f5eb004aacb8f92ed3 fc219ce93262457595eacb4f 5442605311e5452094be8857 c6a25f7ab07e4ec38a58fea2 05c0f3f2f9c1406bb9c7a052 e044b5af826c4b48a48bc4d8 43fe065ab5a7425fbe0ab1d1 94fb3c03cb7b43ac94a093e2 9bb12d4633e74af9a64266d4 a70440160a53423682c04c31 5545638d22ae407fa0e077a2 c8b1440260b346d58bb6adac".split()' \
#   --out_csv=results/plot_rd/tmp.csv \
#   --out_html=results/plot_rd/tmp.html



# run.hp.num_points=1024
# run.hp.num_channels.g_a=[3,16]
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='run.hash in "9447fba58ce04a1b836621d9 ed15eed869c348f08fcbf15b bc7a1998e7384883a5e174b7 8c4a69f9ac2d40fbacedca7c b472c2d6ac764d78a07b13db 7d369f1007c24b97b488097d 1851701e65d2441695331a6a 30198d1b37e04f2387203d11 9fa0733b39cd47aca81e8ace c8a9ddb4348046d3b5ab6bd4 22ed2eb6149a48edaa51b367 c99577da24c148dbbf0066de ba06b075cdd24ce986f33eb5 ec9f1105d16049799e2696d9 bd2516b456cc49319ea05b2c e6c77d1286004fdea99f9b35".split()' \
#   --out_csv=results/plot_rd/tmp.csv \
#   --out_html=results/plot_rd/tmp.html

# # run.hp.num_points=512
# # run.hp.num_channels.g_a=[3,16]
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='run.hash in "bb7486944652489cad3a2c89 1540cae611c1454bba77432e e411bd7be18f4c22b9484d35 c477b2894a7c44e486a4e935 d7d68e9859764a1e8a5423a0 1fe03b2799524037bdf7efe1 44518ee9d7524f1694ec318d bd49ddcb089a4f45ad1d57fe 61f4bae97485469692ecee06 e586ff8ea57347d1b91d3588 f3774aa69055419a8c6bd830 e190f64a15a94aff96383d1d 4dd085864aac41269c3067fe 2e5976382c7e4a7689893c67".split()' \
#   --out_csv=results/plot_rd/tmp.csv \
#   --out_html=results/plot_rd/tmp.html

# # run.hp.num_points=256
# # run.hp.num_channels.g_a=[3,16]
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='run.hash in "76211de7afa841d7b0a5be9a 76f1dd907ccb4365981d43a7 53be35046367458db7688271 5ceb9a61b1ed4e159e089527 ebcbfa6aa950423daba3f4c0 7f2fb2556240470196fb5486 c7769020447c4aaa882f42f6 d8b9e1b2b0a54ec79a9453ae 34e394c444c944d0a0e2f6e7 d7921331f2404114aaaa4084 7c634070046a421c9c88b97a 786fdabbe75246e29567cae9 de9f3213dde64e3d9c32f5f7 ed4d5236a6f4430b9fbf861d c1e48ed1894f40c7bd9c6157 41e6689071c84bd88997c92e".split()' \
#   --out_csv=results/plot_rd/tmp.csv \
#   --out_html=results/plot_rd/tmp.html

# # run.hp.num_points=128
# # run.hp.num_channels.g_a=[3,16]
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='run.hash in "2e178372769f470998d509f3 c3404cb946924c09b53ee55e bec4f1f91c544ac58200d5f3 8df152cffcf34f9ebeb168d2 dad61bbe762e4942bbb3dc67 5dbb5375d08e4d4bbf34e4be feece5ad2cea4a80a476ce8b e736fff7d5354e1581ca97a5 13a24b77d9334277b98d96e8 c2409b9266fc416cad6873c5 f3d6046d4ff748bf978d24d1 72075040a27a4e988e3d1f75 fa94475428754c85abf4b070 a7c7cd62129841fa85edbc40".split()' \
#   --out_csv=results/plot_rd/tmp.csv \
#   --out_html=results/plot_rd/tmp.html

# # run.hp.num_points=64
# # run.hp.num_channels.g_a=[3,16]
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='run.hash in "b022afc6b77d4c64883f6dbc e0c0739731654d48ab70bfe5 c5d81a5b64eb4ca38edc9106 092281fcda184698a5c8c201 469cb0240d3f48a09acb8844 137ad8fe1ff0475da0e04301 8cd7fe909cbc4e04b82eb867 5b36bfe4f7e84cec806d1487 bbfea59c17294e3c8df6f092 3df0169963fe44bcae64819a 7aefc21e1074420c9320b13b e4208cb581f0465ea01fc9e8 fec053ab1c9745dd8bb0754c 843f9f3a0a5f4b8b86f4ae5d".split()' \
#   --out_csv=results/plot_rd/tmp.csv \
#   --out_html=results/plot_rd/tmp.html

# # run.hp.num_points=32
# # run.hp.num_channels.g_a=[3,16]
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='run.hash in "d04b68cfa61c4230bb222161 b1cece584daa4cd6bb7effc6 7c84e98fa332407899149f72 fce64c08291a4590a0eda6af 6da5e18ee6bc436a9042b80d ac8589442b3d46138faf8497 7e9775c39db145a1b0ad09e2 d61b91494caa4e9dae5ac6ab d8f0b5f4197147ddb381b1fb 611eac082787411691aff1f6 f8a59f922bc94527b40047bb 3cc2a96b171e40d7b228e780 083eb08172b44e4387a68fd6 8d36792b9dfb4fe7b026df8b".split()' \
#   --out_csv=results/plot_rd/tmp.csv \
#   --out_html=results/plot_rd/tmp.html

# # run.hp.num_points=16
# # run.hp.num_channels.g_a=[3,16]
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='run.hash in "8f32f06074b84d869d2f9a1b 75db469d4fc64e178db46836 fe3182982785416cb3287490 87da16e6ef39424493df7c1b fc1dd7a797df4561ace00c7c 08f7a4052005453d80bbb83c 6dd0d1d949dc40dab38f46a7 c028ad01b4b141798fc9e8b8 4e58c04043b64742ae290025 0e7516e8e6544601bbc8d0ae 2d9f1076e28646df9bb9e3bf babe2862f56c4890af2492fc b6942869982b4f5dae455d27 26de322da6de4b438cd43171".split()' \
#   --out_csv=results/plot_rd/tmp.csv \
#   --out_html=results/plot_rd/tmp.html

# # run.hp.num_points=8
# # run.hp.num_channels.g_a=[3,16]
# "${PLOT[@]}" "${COMMON_ARGS[@]}" \
#   --query='run.hash in "8c21b43e3fa54b15a6d00963 c67daf2c81b64b75b8315787 daa4c862966d4431907fee4e 9454a35d132748eea67384a3 086cf7bac0044075b028f2d7 0b9c0fb8fbb649d7ac40cb6b bb5d46037d074eddb4a006ba 4af73ebfeb114109bda1bba8 5e8cbe03878a4887a6e8219f 52cf7fa690bd42a6be7aabed e3a5d5d350774faea80fe4dd bf30415799c84855bb461ffd 3797ad5091ff4c7998e7dc80 22ae8d7f6a1c41b78428df9c".split()' \
#   --out_csv=results/plot_rd/tmp.csv \
#   --out_html=results/plot_rd/tmp.html



# Write result JSON files, then plot RD curves:
# # porp scripts/save_json_from_aim_query.py
# # porp scripts/plot_rd_mpl_new.py

# Plot point clouds for "micro" codec:
# # porp scripts/plot_point_cloud_mpl.py aadd008012aa40b7bf41d5e7 d657d2773d784935ac8d373f 87f30aee7f0445a4b3a545d8 1f7de7fd3b274e8b89234bfa f54c39d00c334adf903d6a6e 3fe3d84cd7954b75906961d8 29d47d0a657d47e39e900067 915dbd00bcf04aa2896344cb 6cc7b7899ee7411c9f2dbd51 623023c8ab164652bcdb7386 d85c7e59cab247d790a07038 67be12941e4e427380a03e44; echo "micro"

# # Plot point clouds for "lite" codec:
# porp scripts/plot_point_cloud_mpl.py dd702cc73af74bb6a9b61ff3 409c1460ec784662b14dfcbc 63e1dbf94b694d7693dc2cb7 e783b170eda14aa29224c92b 47f31d8b7ef54d039e8e05d3 3616dc2455214fdeb4f4b62d b459932f207b4063976da932 232b3be0e7934c80beb6e389 76a4b98a929244818f28e1ba 06c3bfe845c44c849780114d d6d13bfbb26a4a9798232a22 7e17d0a21ff340fc95f0da58; echo "lite"

# # Plot point clouds for "full" codec:
# porp scripts/plot_point_cloud_mpl.py fe4122df657d40dea6774a47 b67c87a00d5240cc936f6a43 d2880736c02c44b5b540156c 7a1ccbf555ad4d3e896a9a3b 6e0091c06ce044699ecdd206 19f2d7a78f1740f18decd4c9 7c69baf921d84a4482dd1be6 92088f9972e4495aaeefb770 35482eca40bc4d88b3598894 677b6063456b4930a27c2857 2e2eb122c87a4041b7f3f587 5f2ab297622a4a3fa3a1214b; echo "full"



# # Query relevant run_hashes.
#
# poetry run python -c '
# import sys
# import aim
# from compressai_trainer.utils.aim.query import get_runs_dataframe, run_hashes_by_query
# repo = aim.Repo(sys.argv[1])
# query = sys.argv[2]
# run_hashes = run_hashes_by_query(repo, query)
# df = get_runs_dataframe(run_hashes, repo=repo, metrics=["bpp_loss", "acc_top1"], hparams=["criterion.lmbda.cls", "hp.num_points", "hp.num_channels.g_a"])
# df["hp.num_channels.g_a"] = df["hp.num_channels.g_a"].map(tuple)
# df = df.sort_values(["hp.num_channels.g_a", "model.name", "criterion.lmbda.cls"]).reset_index(drop=True)
# df = df[["run_hash", "hp.num_channels.g_a", "model.name", "criterion.lmbda.cls", "bpp_loss", "acc_top1"]]
# print(df.to_string(index=False))
# ' \
#   "$HOME/data/aim/pc-mordor/pcc" \
#   "run.hp.num_points == 1024"


# porp -c 'import sys; import aim; from compressai_trainer.utils.aim.query import get_runs_dataframe, run_hashes_by_query; repo = aim.Repo(sys.argv[1]); query = sys.argv[2]; run_hashes = run_hashes_by_query(repo, query); df = get_runs_dataframe(run_hashes, repo=repo, metrics=["bpp_loss", "acc_top1"], hparams=["criterion.lmbda.cls", "hp.num_points", "hp.num_channels.g_a"]); df["hp.num_channels.g_a"] = df["hp.num_channels.g_a"].map(tuple); df = df.sort_values(["hp.num_channels.g_a", "model.name", "criterion.lmbda.cls"]).reset_index(drop=True); df = df[["run_hash", "hp.num_channels.g_a", "model.name", "criterion.lmbda.cls", "bpp_loss", "acc_top1"]]; print(df.to_string(index=False))' "$HOME/data/aim/pc-mordor/pcc" "run.hp.num_points == 1024"



# CRITICAL POINT SETS

# Classification-only:
#
# RUN_HASHES=(
#   # micro
#   # 9447fba58ce04a1b836621d9  # um-pcc-cls-only-pointnet-mini-001  10.620185  0.528363                   10
#   ed15eed869c348f08fcbf15b  # um-pcc-cls-only-pointnet-mini-001  12.647167  0.600081                   14
#   # bc7a1998e7384883a5e174b7  # um-pcc-cls-only-pointnet-mini-001  16.284716  0.654376                   20
#   # 8c4a69f9ac2d40fbacedca7c  # um-pcc-cls-only-pointnet-mini-001  20.186681  0.707861                   28
#   b472c2d6ac764d78a07b13db  # um-pcc-cls-only-pointnet-mini-001  24.557064  0.742707                   40
#   # 7d369f1007c24b97b488097d  # um-pcc-cls-only-pointnet-mini-001  36.497228  0.775932                   80
#   1851701e65d2441695331a6a  # um-pcc-cls-only-pointnet-mini-001  48.696047  0.805511                  160
#   # 30198d1b37e04f2387203d11  # um-pcc-cls-only-pointnet-mini-001  61.583964  0.816856                  320
#   # 9fa0733b39cd47aca81e8ace  # um-pcc-cls-only-pointnet-mini-001  73.351176  0.824554                 1000
#   # c8a9ddb4348046d3b5ab6bd4  # um-pcc-cls-only-pointnet-mini-001  78.305946  0.825770                 4000
#   # 22ed2eb6149a48edaa51b367  # um-pcc-cls-only-pointnet-mini-001  78.062356  0.827391                 6000
#   ec9f1105d16049799e2696d9  # um-pcc-cls-only-pointnet-mini-001  78.978195  0.829417                16000
#   # c99577da24c148dbbf0066de  # um-pcc-cls-only-pointnet-mini-001  78.144314  0.821313               164000
#
#   # lite
#   ea428fd42da84dbcbc1f7c34  # um-pcc-cls-only-pointnet           10.983490  0.579822                   10
#   5e90b82ce0ab44808345d57b  # um-pcc-cls-only-pointnet           12.454252  0.636143                   14
#   ae645b27cd59429193667fb5  # um-pcc-cls-only-pointnet           13.723202  0.668963                   20
#   95c1ea383a584e5aa29d63ce  # um-pcc-cls-only-pointnet           18.090725  0.719206                   28
#   32d71c950b6243a5b3bc322d  # um-pcc-cls-only-pointnet           22.008024  0.762966                   40
#   b159f611678f405fb942b8e7  # um-pcc-cls-only-pointnet           32.941536  0.794165                   80
#   0b27e0776f534cf485cf56e6  # um-pcc-cls-only-pointnet           48.020104  0.820502                  160
#   7189f08374f14026a1d782d5  # um-pcc-cls-only-pointnet           65.661915  0.844408                  320
#   # 73528fbba3624827823566bd  # um-pcc-cls-only-pointnet          105.422323  0.845624                 1000
#   a7dceeefa39747ed90bf9e47  # um-pcc-cls-only-pointnet          137.008761  0.850486                 4000
#   # cdbb97f57a264f80acf73543  # um-pcc-cls-only-pointnet          157.043000  0.847245                16000
#   # 7ab67b4d6fab45a4bc2cf9a3  # um-pcc-cls-only-pointnet          153.931869  0.841977                64000
#   # eff4d7bce95c4c1582107a45  # um-pcc-cls-only-pointnet          160.215024  0.848460               256000
#
#   # full
#   8a59f52ab3d04649913357c3  # um-pcc-cls-only-pointnet            8.055724  0.451378                   10
#   48b76524474e4c1b8206056d  # um-pcc-cls-only-pointnet           10.378093  0.593598                   14
#   18388dd3556b4af8ba48c038  # um-pcc-cls-only-pointnet           16.711520  0.724878                   28
#   55f1acfe0e304065b328202b  # um-pcc-cls-only-pointnet           14.498847  0.694895                   20
#   5d374263257a4f08b73bf21d  # um-pcc-cls-only-pointnet           20.964733  0.765397                   40
#   11b05bf4293e4e728caf66a1  # um-pcc-cls-only-pointnet           30.778704  0.810778                   80
#   c9e04f474c2b4852a4ddaf85  # um-pcc-cls-only-pointnet           46.490746  0.839546                  160
#   dfd71410fa16468697fb2d47  # um-pcc-cls-only-pointnet           60.244190  0.850081                  240
#   dc57d83678854c2eaf63ac5e  # um-pcc-cls-only-pointnet           74.491030  0.858995                  320
#   817ef4ee42c04083a3fbed82  # um-pcc-cls-only-pointnet          108.516576  0.863047                  640
#   deff251f04694d46ac04b797  # um-pcc-cls-only-pointnet          139.423393  0.876418                 1000
#   # 3ff11150c09c437e9b415df4  # um-pcc-cls-only-pointnet          208.790239  0.876823                 2000
#   # 80cc133d0b8847bab5dd7f03  # um-pcc-cls-only-pointnet          317.559084  0.878444                 4000
#   d701c67e3f03430e930ddadd  # um-pcc-cls-only-pointnet          789.780310  0.884927                16000
#   # ffe35513aed94664bcffbb57  # um-pcc-cls-only-pointnet         2251.736108  0.882901                64000
#   # 05ec3dc03d054eeb80a313c2  # um-pcc-cls-only-pointnet         3730.229713  0.879660               256000
#   # 791e48be414f49d5ab9f922b  # um-pcc-cls-only-pointnet         4154.949795  0.883712               512000
#   # 9e514bff8b574694ace3d18a  # um-pcc-cls-only-pointnet         4244.303898  0.875203              1024000
#   # d3ecd54998414437a956dc59  # um-pcc-cls-only-pointnet         4328.507145  0.879660              2048000
#   # 1571d65ecb63423c975e8ac9  # um-pcc-cls-only-pointnet         4353.796289  0.878849              4096000
#   # 37810e597237441caa9e8a33  # um-pcc-cls-only-pointnet         4455.765900  0.877634             16384000
#   # b5fb04685dbe4e748f294c88  # um-pcc-cls-only-pointnet         4531.615613  0.881280             65536000
# )


RUN_HASHES=(
#                 run_hash  # hp.num_channels.g_a                        model.name  criterion.lmbda.cls    bpp_loss  acc_top1
# 9447fba58ce04a1b836621d9  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                   10   10.620185  0.528363
# ed15eed869c348f08fcbf15b  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                   14   12.647167  0.600081
# bc7a1998e7384883a5e174b7  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                   20   16.284716  0.654376
# 8c4a69f9ac2d40fbacedca7c  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                   28   20.186681  0.707861
# b472c2d6ac764d78a07b13db  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                   40   24.557064  0.742707
# 7d369f1007c24b97b488097d  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                   80   36.497228  0.775932
# 1851701e65d2441695331a6a  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                  160   48.696047  0.805511
# 30198d1b37e04f2387203d11  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                  320   61.583964  0.816856
# 9fa0733b39cd47aca81e8ace  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                 1000   73.351176  0.824554
# c8a9ddb4348046d3b5ab6bd4  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                 4000   78.305946  0.825770
# 22ed2eb6149a48edaa51b367  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                 6000   78.062356  0.827391
# ba06b075cdd24ce986f33eb5  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                 8000   79.214763  0.819287
# 38b6b528ab2c408d9e1f6fcd  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                16000   78.999304  0.830227
# 75e587404be44441bb8b92c5  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                16000   77.732358  0.826175
# ec9f1105d16049799e2696d9  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                16000   78.978195  0.829417
# bd2516b456cc49319ea05b2c  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001                64000   78.089627  0.818071
# c99577da24c148dbbf0066de  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001               164000   78.144314  0.821313
# e6c77d1286004fdea99f9b35  #                       (3, 16) um-pcc-cls-only-pointnet-mini-001               256000   78.294346  0.820097
# aadd008012aa40b7bf41d5e7  #                       (3, 16)     um-pcc-multitask-cls-pointnet                   10   43.530414  0.733793
  d657d2773d784935ac8d373f  #                       (3, 16)     um-pcc-multitask-cls-pointnet                   14   31.798217  0.684360
# 87f30aee7f0445a4b3a545d8  #                       (3, 16)     um-pcc-multitask-cls-pointnet                   20   42.148722  0.745948
  1f7de7fd3b274e8b89234bfa  #                       (3, 16)     um-pcc-multitask-cls-pointnet                   28   20.582290  0.711102
  f54c39d00c334adf903d6a6e  #                       (3, 16)     um-pcc-multitask-cls-pointnet                   40   28.714629  0.738655
# 3fe3d84cd7954b75906961d8  #                       (3, 16)     um-pcc-multitask-cls-pointnet                   80   41.712438  0.797002
# 29d47d0a657d47e39e900067  #                       (3, 16)     um-pcc-multitask-cls-pointnet                  160   54.002624  0.802269
# 915dbd00bcf04aa2896344cb  #                       (3, 16)     um-pcc-multitask-cls-pointnet                  320   68.545377  0.815640
  6cc7b7899ee7411c9f2dbd51  #                       (3, 16)     um-pcc-multitask-cls-pointnet                 1000   76.265409  0.820502
# 623023c8ab164652bcdb7386  #                       (3, 16)     um-pcc-multitask-cls-pointnet                 4000   77.950269  0.817666
# d85c7e59cab247d790a07038  #                       (3, 16)     um-pcc-multitask-cls-pointnet                 8000   78.470701  0.821718
# 9d1c625213474253989bb599  #                       (3, 16)     um-pcc-multitask-cls-pointnet                16000   76.367574  0.816045  # NOTE What happened here w.r.t. airplane sample model?! WOW! Just became a blob... And yet, the average accuracy...
# 370e286fbaf24ecf85850caf  #                       (3, 16)     um-pcc-multitask-cls-pointnet                16000    0.076509  0.820908
# 67be12941e4e427380a03e44  #                       (3, 16)     um-pcc-multitask-cls-pointnet                16000   78.049098  0.821313
# 25db4c30a1fe43ff93fbf0f6  #                       (3, 16)     um-pcc-multitask-cls-pointnet                16000   77.748679  0.824554
# ea428fd42da84dbcbc1f7c34  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                   10   10.983490  0.579822
# 5e90b82ce0ab44808345d57b  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                   14   12.454252  0.636143
# ae645b27cd59429193667fb5  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                   20   13.723202  0.668963
# 95c1ea383a584e5aa29d63ce  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                   28   18.090725  0.719206
# 32d71c950b6243a5b3bc322d  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                   40   22.008024  0.762966
# b159f611678f405fb942b8e7  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                   80   32.941536  0.794165
# 0b27e0776f534cf485cf56e6  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                  160   48.020104  0.820502
# 7189f08374f14026a1d782d5  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                  320   65.661915  0.844408
# 73528fbba3624827823566bd  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                 1000  105.422323  0.845624
# a7dceeefa39747ed90bf9e47  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                 4000  137.008761  0.850486
# cdbb97f57a264f80acf73543  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                16000  157.043000  0.847245
# 7ab67b4d6fab45a4bc2cf9a3  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet                64000  153.931869  0.841977
# eff4d7bce95c4c1582107a45  #         (3, 8, 8, 16, 16, 32)          um-pcc-cls-only-pointnet               256000  160.215024  0.848460
# dd702cc73af74bb6a9b61ff3  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                   10   97.743256  0.717180
  409c1460ec784662b14dfcbc  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                   14   15.061701  0.619935
  63e1dbf94b694d7693dc2cb7  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                   20   18.964734  0.695705
# e783b170eda14aa29224c92b  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                   28   24.115921  0.705024
# 47f31d8b7ef54d039e8e05d3  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                   40   26.246635  0.778768
  3616dc2455214fdeb4f4b62d  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                   80   39.130638  0.809968
# b459932f207b4063976da932  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                  160   56.186260  0.832253
# 232b3be0e7934c80beb6e389  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                  320   73.038072  0.834684
  76a4b98a929244818f28e1ba  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                 1000  130.053458  0.841977
# 06c3bfe845c44c849780114d  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                 4000  143.230317  0.841977
# d6d13bfbb26a4a9798232a22  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                 8000  147.347270  0.842382
# 9ca415935ee64e2fa1dfeb4f  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                16000  148.434026  0.841572
# 7e17d0a21ff340fc95f0da58  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                16000  149.592830  0.848460
# 8a59f52ab3d04649913357c3  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                   10    8.055724  0.451378
# 48b76524474e4c1b8206056d  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                   14   10.378093  0.593598
# 55f1acfe0e304065b328202b  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                   20   14.498847  0.694895
# 18388dd3556b4af8ba48c038  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                   28   16.711520  0.724878
# 5d374263257a4f08b73bf21d  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                   40   20.964733  0.765397
# 11b05bf4293e4e728caf66a1  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                   80   30.778704  0.810778
# c9e04f474c2b4852a4ddaf85  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                  160   46.490746  0.839546
# dfd71410fa16468697fb2d47  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                  240   60.244190  0.850081
# dc57d83678854c2eaf63ac5e  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                  320   74.491030  0.858995
# 817ef4ee42c04083a3fbed82  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                  640  108.516576  0.863047
# deff251f04694d46ac04b797  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                 1000  139.423393  0.876418
# 3ff11150c09c437e9b415df4  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                 2000  208.790239  0.876823
# 80cc133d0b8847bab5dd7f03  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                 4000  317.559084  0.878444
# d701c67e3f03430e930ddadd  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                16000  789.780310  0.884927
# ffe35513aed94664bcffbb57  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet                64000 2251.736108  0.882901
# 05ec3dc03d054eeb80a313c2  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet               256000 3730.229713  0.879660
# 791e48be414f49d5ab9f922b  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet               512000 4154.949795  0.883712
# 9e514bff8b574694ace3d18a  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet              1024000 4244.303898  0.875203
# d3ecd54998414437a956dc59  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet              2048000 4328.507145  0.879660
# 1571d65ecb63423c975e8ac9  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet              4096000 4353.796289  0.878849
# 37810e597237441caa9e8a33  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet             16384000 4455.765900  0.877634
# b5fb04685dbe4e748f294c88  #    (3, 64, 64, 64, 128, 1024)          um-pcc-cls-only-pointnet             65536000 4531.615613  0.881280
# fe4122df657d40dea6774a47  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                   10   11.220641  0.560778
  b67c87a00d5240cc936f6a43  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                   14   15.212449  0.659238
# d2880736c02c44b5b540156c  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                   20   17.641623  0.725284  # NOTE also interesting...
  7a1ccbf555ad4d3e896a9a3b  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                   28   21.752992  0.777147
# 6e0091c06ce044699ecdd206  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                   40   25.137637  0.788898
# 19f2d7a78f1740f18decd4c9  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                   80   35.037599  0.836305
  7c69baf921d84a4482dd1be6  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                  160   53.775588  0.852512
# 92088f9972e4495aaeefb770  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                  320   79.064011  0.865478
# 35482eca40bc4d88b3598894  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                 1000  172.812956  0.866288
  677b6063456b4930a27c2857  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                 4000  347.090900  0.880875
# 2e2eb122c87a4041b7f3f587  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                 8000  551.711173  0.880875
# 5f2ab297622a4a3fa3a1214b  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                16000  696.513746  0.877634
# bf7e8d76f85f423bb41c7158  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                16000  891.239041  0.874392
# 2b60e7ccebd84e01a121abba  #                       (3, 32) um-pcc-cls-only-pointnet-mini-001                16000  144.800634  0.814830
# 2ad75fda850a46f4926648ff  #                       (3, 33) um-pcc-cls-only-pointnet-mini-001                16000  150.874226  0.831037
# f738dd2f866244189101c050  #                       (3, 33) um-pcc-cls-only-pointnet-mini-001                16000  160.415925  0.716775
# 04827a25c49242ccbc6c90d0  #                       (3, 66) um-pcc-cls-only-pointnet-mini-001                16000  307.946254  0.710697
# e33502d57812453aa21b6e79  #                    (3, 4, 32) um-pcc-cls-only-pointnet-mini-001                16000   86.801987  0.601702
# dd6aeb4cddd14d8192766551  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                   10    9.296978  0.543760
# df4e180abb2241d4880a6bde  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                   14   12.657459  0.622771
# ca94229306a04e2e8a723711  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                   20   15.098439  0.674230
# a8a20cb8cc8547d39519d665  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                   28   18.968165  0.721232
# 988d55d3399c4877bdec3571  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                   40   22.927232  0.769854
# 59d2aa10d2c343f7ab95b7c7  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                   80   30.493327  0.782010
# d873d0d114cf4f0db6816177  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                  160   46.007315  0.800243
# 61e9fd7a8bdd4a219fdba50d  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                  160  152.054280  0.606969
# f2a021eebb3d405a950b79af  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                  160   43.756137  0.816451
# 7acb6b78c3e744d2bb624343  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                  160   47.387769  0.806321
# 2992f894e8a64822ba0d2188  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                  160   50.461308  0.821313
# 1c881125f138443493e8ba16  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                  160   48.572602  0.807942
# c651b7f2e2cd4c59a50de281  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                  160   47.992585  0.809157
# 9f8724fb326a4ef4bbbc13d2  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                  320   69.187710  0.825770
# cd8cb67a0e32483483cfc48c  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                 1000  111.212405  0.831442
# 79fbc97a56ad4bd390da424e  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                 4000  136.483891  0.827391
# 1e90542553204cc7b9945b22  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                16000  141.078980  0.831037
# befef72bec9b437e87fb9700  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                16000  142.281192  0.830632
# 66930b25010a4f009815f88c  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                16000  141.897184  0.825365
# 192718b578434d4797d713e7  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                16000  144.258711  0.835900
# 8b0601cf0fd1468faf47ca46  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                16000  147.325375  0.822123
# 36f625472fa448f5830f568f  #                 (3, 8, 8, 32) um-pcc-cls-only-pointnet-mini-001                16000  120.916641  0.831037
# a1064758a50e461690532669  #                 (3, 8, 8, 64) um-pcc-cls-only-pointnet-mini-001                16000  213.917874  0.831442
# c699ec6f3c2e4d77b1d8d15c  #                (3, 8, 16, 64) um-pcc-cls-only-pointnet-mini-001                16000  209.282557  0.832658
# 5c6752c4cb9d4acfbee11717  #                    (3, 8, 32) um-pcc-cls-only-pointnet-mini-001                  160   45.781502  0.796596
# b917b156b7f741b88c33d3f5  #                    (3, 8, 32) um-pcc-cls-only-pointnet-mini-001                16000  113.622243  0.816045
# cce37f2f54584a5c9266cc08  #                    (3, 8, 32) um-pcc-cls-only-pointnet-mini-001                16000  114.585639  0.816451
# 9e12dd9929fe4a02adfa849d  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                   10    9.905030  0.478930
# d43385b8f039447d85c78171  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                   14   12.407139  0.567261
# 98c5ac61d374472e83cd2b8e  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                   16   13.248549  0.591167
# 574386ebe91a4571a88f558f  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                   20   15.633077  0.631686
# b68f7630627e4238b40bcc2f  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                   28   19.888383  0.701378
# 61471e16d91f4b24bd2a9c8e  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                   40   23.820020  0.722853
# dc54ca5b5c0c4a52ba00da0e  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                   80   31.886107  0.768639
# f68f2282166b40dc977e20bd  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                  160   46.423074  0.799838
# 023ff4ecb494450ab2c336dd  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                  160   51.443622  0.803079
# b63dcdbd7ba5459995b2212d  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                  320   69.010982  0.826175
# d3756ef84fa246f4846c58a0  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                 1000  126.716031  0.826580
# 4c1510780fe94923aaeba1ef  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                 4000  211.315149  0.827796
# 50e59b3e01294da391d6e212  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                 6000  212.664158  0.838331
# e36643a73ce44f98b7306f63  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                16000  228.017627  0.831848
# 38c0ebd176874536a7474c0b  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                16000  149.534167  0.745948
# 538989d7b0524bdf9978620e  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                16000  223.183946  0.822934
# 2d6366fa08ac4a5090511c25  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001                16000  228.017627  0.831848
# aca683c052df45c5a44d1f48  #                    (3, 8, 64) um-pcc-cls-only-pointnet-mini-001               164000  236.782315  0.836305
)

# Condensed version:
RUN_HASHES=(
#                 run_hash  # hp.num_channels.g_a                        model.name  criterion.lmbda.cls    bpp_loss  acc_top1
  d657d2773d784935ac8d373f  #                       (3, 16)     um-pcc-multitask-cls-pointnet                   14   31.798217  0.684360
  1f7de7fd3b274e8b89234bfa  #                       (3, 16)     um-pcc-multitask-cls-pointnet                   28   20.582290  0.711102
  f54c39d00c334adf903d6a6e  #                       (3, 16)     um-pcc-multitask-cls-pointnet                   40   28.714629  0.738655
  6cc7b7899ee7411c9f2dbd51  #                       (3, 16)     um-pcc-multitask-cls-pointnet                 1000   76.265409  0.820502
  409c1460ec784662b14dfcbc  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                   14   15.061701  0.619935
  63e1dbf94b694d7693dc2cb7  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                   20   18.964734  0.695705
  3616dc2455214fdeb4f4b62d  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                   80   39.130638  0.809968
  76a4b98a929244818f28e1ba  #         (3, 8, 8, 16, 16, 32)     um-pcc-multitask-cls-pointnet                 1000  130.053458  0.841977
  b67c87a00d5240cc936f6a43  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                   14   15.212449  0.659238
  7a1ccbf555ad4d3e896a9a3b  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                   28   21.752992  0.777147
  7c69baf921d84a4482dd1be6  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                  160   53.775588  0.852512
  677b6063456b4930a27c2857  #    (3, 64, 64, 64, 128, 1024)     um-pcc-multitask-cls-pointnet                 4000  347.090900  0.880875
)

FILENAMES=(
  micro_1
  micro_2
  micro_3
  micro_4
  lite_1
  lite_2
  lite_3
  lite_4
  full_1
  full_2
  full_3
  full_4
)

mkdir -p results/plot_pointcloud/pdf/crit/
mkdir -p results/plot_pointcloud/pdf/rec/

for ((i=0; i < "${#RUN_HASHES[@]}"; i+=1)); do
  run_hash="${RUN_HASHES[i]}"
  filename="${FILENAMES[i]}"
  echo ">>> ${i} ${run_hash} ${filename}"
  poetry run python scripts/plot_critical_point_set.py \
    --config-path="$HOME/data/runs/pc-mordor/${run_hash}/configs" \
    --config-name="config" \
    ++model.source="config" \
    ++paths.model_checkpoint='${paths.checkpoints}/runner.last.pth' \
    ++misc.out_path.critical="results/plot_pointcloud/pdf/crit/${filename}.pdf" \
    ++misc.out_path.reconstruction="results/plot_pointcloud/pdf/rec/${filename}.pdf"
    # ++misc.out_path.critical="results/plot_pointcloud/pdf/crit/${i}_${run_hash}.pdf" \
    # ++misc.out_path.reconstruction="results/plot_pointcloud/pdf/rec/${i}_${run_hash}.pdf"
done


