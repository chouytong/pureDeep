# 复核与重现

全部在服务器/home/zyt/deep_final，Python /home/zyt/envs/mfam_analysis/bin/python -B。依赖版本见audit/environment.json，不安装XGBoost。主线起点35b0a7c，15canonical splits×seeds42/43/44。

在独立且具相同私有资产的环境依次执行scripts/audit8.py、ensemble8.py、readout8.py、interaction8.py、supcon8.py、comparison8.py、benchmark8.py、plot8.py、report8.py、final_audit8.py。各阶段freeze/output防覆盖，不得在现有目录直接重跑；新环境应把HERE设为一个全新隔离目录，先验证基线与私有资产哈希。新source路径hash相应改变应明确记录，不能伪造原freeze。plot8.py使用已有隔离report依赖路径，不安装包。

公开仓库不含数据、逐人缓存、权重与私有ID manifests，因此只凭公开仓库不能完整复训。这是访问限制，非声称数据已发布。表格均由本目录脚本生成；正式报告入口reports/FINAL_REPORT.md；原始stage报告和source冻结保留。
