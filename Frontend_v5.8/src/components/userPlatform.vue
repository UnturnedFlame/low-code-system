<template>
  <!--
  更新页面 --v5.6
  版本  2024-7-19
  -->
  <div style="height: 100vh; overflow:hidden" @mouseover="background_IMG">
    <el-container class="fullscreen_container">
      <el-header style="height: 60px;text-align: center; line-height: 60px; position: relative;">
        <img src="../assets/system-logo.png" alt="" style="width: 130px; height: auto; position: absolute; left: 5px; top: 5px; color: white;">
        <h2 style="font-size: 26px;">车轮状态分析与健康评估软件</h2>   
        <div class="user-info-container" id="userInfo" style="position: absolute; right: 10px; top: 5px; color: white;">
          <a-dropdown :trigger="['click']" class="clickable" placement="bottomLeft">
            <a  @click.prevent>
              帮助
            </a>
            <template #overlay>
              <a-menu>
                <a-menu-item @click="operationHelpDialogVisible=true">
                  <!-- <span @click="operationHelpDialogVisible=true">操作指南</span> -->
                  操作指南
                </a-menu-item>
                <a-menu-item @click="userHelpDialogVisible=true">
                  <!-- <span @click="userHelpDialogVisible=true">使用教程</span> -->
                  使用教程
                </a-menu-item>
              </a-menu>
            </template>
          </a-dropdown>
          
          <span style="margin-right: 18px; margin-left: 18px">欢迎！ {{ username }}</span>
          <span @click="logout" class="clickable">退出登录</span>
          
          
          <!-- <span class="clickable" style="margin-left: 10px" @click="helpDialogVisible=true">帮助</span> -->
        </div>
        <!-- 操作指南 -->
        <el-dialog v-model="operationHelpDialogVisible" title="操作指南" width="810" draggable :close-on-click-modal="false" :center="false">
          <div style="text-align: left;">
            <el-scrollbar height="500px">
              <h1>1、选择算法</h1>
              <h3>从算法选择区中将所选择算法模块拖拽至可视化建模区并调整位置</h3>
              <img src="../assets/step_1.gif" alt="" style="width: 700px; height: auto">
              <h1>2、调整参数</h1>
              <h3>鼠标右击可视化建模区中的算法模块，在弹出的下拉框中更改算法的参数</h3>
              <img src="../assets/step_2.gif" alt="" style="width: 700px; height: auto">
              <h1>3、建立流程</h1>
              <h3>鼠标移至作为所建立流程的起点的算法模块上的红色附着点，点击并拖拽至所建立流程的目标算法模块</h3>
              <img src="../assets/step_3.gif" alt="" style="width: 700px; height: auto">
              <h1>4、检查模型及修改模型</h1>
              <h3>在确保所有模块都已经建立流程后，点击完成建模，然后点击检查模型对所建立的模型进行检查，</h3>
              <h3>如果模型中存在错误，点击修改模型并根据提示对模型进行修改，然后依次点击完成建模和模型检查，</h3>
              <h3>通过模型检查后即可以保存模型，并进行后续操作。</h3>
              <img src="../assets/step_4.gif" alt="" style="width: 750px; height: auto">
              
            </el-scrollbar>
          </div>
          
        </el-dialog>
        <!-- 使用教程 -->
        <el-dialog v-model="userHelpDialogVisible" title="使用教程" width="810" draggable :close-on-click-modal="false" :center="false">
          <div style="text-align: left;">
            <el-scrollbar height="500px" ref="userHelpDialogScrollbar">
              <h2>常见问题</h2>
               
              <div id="howToUseThisApp">
                <h2>1、如何使用本软件？</h2>
                <h3>本软件分为五个部分：①算法选择区、②数据加载区、③可视化建模区、④结果可视化区、⑤加载模型区。</h3>
                <img src="../assets/system-outline.png" alt="" style="width: 700px; height: auto">
                <h3>①<span style="color: red;">数据加载区</span>用于上传数据文件，查看用户的历史文件以及加载数据文件。</h3>
                <h3>②<span style="color: red;">算法选择区</span>中包含本系统支持的算法模块，可以选择其中的算法模块并拖入到可视化建模区进行建模。</h3>
                <h3>③<span style="color: red;">可视化建模区</span>是进行建模的区域，其中包含所有已经建立流程的算法模块，可以拖拽模块进行位置调整，也可以右击模块进行参数调整。</h3>
                <h3>④<span style="color: red;">结果可视化区</span>用于查看模型运行结果，包括模型运行结果图、模型运行结果表格。</h3>
                <h3>⑤<span style="color: red;">加载模型区</span>用于查看用户曾经保存的历史模型，和加载已保存的模型。</h3>
              </div>
              
              <div>
                <h2>2、如何在算法选择区中选择对应算法？</h2>
                <h3>在算法选择区中，每个模块都有三级展开结构，</h3>
                <img src="../assets/algorithms-unfold.png" alt="" style="width: 300px; height: auto">
                <h3>点击对应模块展开到最后一级，这时再点击对应算法会在结果可视化区中呈现该算法的介绍。</h3>
                <img src="../assets/algorithm-introduction.png" alt="" style="width: 700px; height: auto">
                <h3>通过拖动的方式可以将指定的算法模块拖入可视化建模区。并可以通过点击右键调整算法参数。</h3>
                <img src="../assets/set-params.png" alt="" style="width: 700px; height: auto">

              </div>

              <div>
                <h2>3、如何建立模型并保存？</h2>
                <h3>在算法选择区中选择对应算法拖入到可视化建模区后，通过点击可视化建模区中算法模块上右侧的红色附着点，可以拉取一条连线并可连接到另一个算法模块上，以此来表示模型的运行顺序。</h3>
                <img src="../assets/line.gif" alt="" style="width: 700px; height: auto">
                <h3>在建立好模型后，点击完成建模，此时可以点击检查模型进行模型检查，具体操作步骤如下。</h3>
                <h3>第一步，点击完成建模</h3>
                <img src="../assets/modeling-finish-1.png" alt="" style="width: 700px; height: auto">
                <h3>第二步，点击检查模型。如果模型中存在错误，会呈现错误提示。</h3>
                <img src="../assets/modeling-finish-2.png" alt="" style="width: 700px; height: auto">
                <img src="../assets/check-model-tip.png" alt="" style="width: 700px; height: auto">
                
                <h3>第三步，如果模型中存在错误，点击修改模型并根据提示对模型进行修改，</h3>
                <img src="../assets/rectify-model-1.png" alt="" style="width: 700px; height: auto">
                <h3>点击修改模型，此时提示正在修改模型，标红的连线表示该处存在逻辑错误，可以点击模块右上方红色的删除按钮删除报错模块，</h3>
                <img src="../assets/rectify-model-2.png" alt="" style="width: 700px; height: auto">
                <h3>删除报错模块后，正确建立模型流程，然后重复上述过程，直到通过模型检查，</h3>
                <img src="../assets/rectify-model-3.png" alt="" style="width: 700px; height: auto">

                <h3>第四步，完成上述流程后，点击保存模型进行模型的保存，</h3>
                <img src="../assets/save-model-1.png" alt="" style="width: 700px; height: auto">
                <h3>输入模型名称，点击确定，</h3>
                <img src="../assets/save-model-2.png" alt="" style="width: 700px; height: auto">

                <h3>其中建模时的模型检查遵循如下流程图中的规则，</h3>
                <img src="../assets/modeling-processing.png" alt="" style="width: 300px; height: auto">
                
              </div>

              <div>
                <h2>4、如何查看历史模型？</h2>
                <h3>在加载模型区中，点击用户历史模型，</h3>
                <img src="../assets/browser-saved-models-1.png" alt="" style="width: 700px; height: auto">
                <h3>此时左侧弹窗中显示的就是用户保存过的历史模型，并且可以点击使用复现历史模型，点击删除历史模型，或是点击查看历史模型信息，</h3>
                <img src="../assets/browser-saved-models-2.png" alt="" style="width: 700px; height: auto">
                </img>
              </div>

              <div>
                <h2>5、如何上传文件到服务器？</h2>
                <h3>第一步，在数据加载区中，点击本地文件，</h3>
                <img src="../assets/upload-data-1.png" alt="" style="width: 700px; height: auto">
              
                <h3>点击选择文件，选择本地文件，使其加载到文件列表中，每次只可以上传一个文件，并且可以点击文件列表中的删除图标要上传的文件，</h3>
                <img src="../assets/upload-data-3.png" alt="" style="width: 300px; height: auto">
                <h3>第二步，点击上传至服务器，根据提示输入文件名与文件描述，点击确定进行上传</h3>
                <img src="../assets/upload-data-2.png" alt="" style="width: 700px; height: auto">
              </div>

              <!-- <a href="javascript:void(0);" @click="scrollTo('howToUseThisApp')">1、如何使用本软件？</a>  -->
              
            </el-scrollbar>
          </div>
        </el-dialog>
      </el-header>
      <el-container>
        <el-aside width="250px">
          <!-- #80a5ba -->
          <div style="font-size: 20px; font-weight: 700; background-color: #1F5EBA; width: 250px; color: #f9fbfa;">
            算法选择区</div>
          <!-- #eff3f6 -->
          <div style="background-color: #FCFCFD; width: 250px;height: 500px;">
            <el-scrollbar height="500px" min-size="35" style="margin-left: 10px;">
              <el-column v-for="item in menuList2">
                <!-- #4599be #5A87F8 -->
                <el-row><el-button style="width: 150px; margin-top: 10px; background-color: #2869C7; color: white; "
                    icon="ArrowDown" @click="menuDetailsSecond[item.label] = !menuDetailsSecond[item.label]">
                    <el-text style="width: 105px; font-size: 15px; color: white;" truncated>{{ item.label
                      }}</el-text></el-button></el-row>

                <el-column v-if="menuDetailsSecond[item.label]" v-for="option in item.options">

                  <el-row style="margin-left: 20px;">
                    <!--  #75acc3 -->
                    <el-button style="width: 150px; margin-top: 7px; background-color: #4A81D3;" icon="ArrowDown"
                      type="info" @click="menuDetailsThird[option.label] = !menuDetailsThird[option.label]">
                      <el-text style="width: 105px; font-size: 12px; color: white;" truncated>{{ option.label
                        }}</el-text></el-button>
                  </el-row>
                  <el-column v-if="menuDetailsThird[option.label]"
                    v-for="algorithm in Object.keys(option.parameters)">
                    <el-tooltip placement="right-start" :content="labelsForAlgorithms[algorithm]" effect="light">
                      <!-- #f9fcff -->
                      <div :draggable="true" @dragend="handleDragend($event, algorithm, option)" class="item"
                        @click="showIntroduction(algorithm.replace(/_multiple/g, ''))"
                        style="background-color: #7BA0D5 ; margin-top: 7px; width: 145px; height: 30px; margin-bottom: 10px; padding: 0px; border-radius: 5px; align-content: center; margin-left: 40px;">
                        <el-text style="width: 105px; font-size: 12px; color: white;" truncated>{{ labelsForAlgorithms[algorithm]
                          }}</el-text>
                      </div>
                    </el-tooltip>
                  </el-column>
                </el-column>
              </el-column>
            </el-scrollbar>

          </div>
          <div class="aside-title">
            加载数据
          </div>
          <div style="width: 250px; height: 180px; position: relative; background-color: white;">
            <uploadDatafile @switchDrawer="handleSwitchDrawer" :api="api" />
            <div style=" width: 250px; height: 20px; position: absolute; left: 5px; top: 140px">已加载数据：{{ usingDatafile }}</div>
          </div>
          
          <div class="aside-title">
            加载模型
          </div>
          <div style="position: relative; width: 250px; height: 250px; background-color: #FCFDFF;">
            <a-button style="width: 165px; font-size: 16px; position:absolute; top: 25px; left: 40px; text-align: center; background-color: #2082F9; color: white;"
              @click="fetchModels">
              <!-- <template #prefix>
                <div>
                  你好
                </div>
              </template> -->
              用户历史模型
            </a-button>
            <div style="position:absolute; top: 65px; left: 5px; width: 240px; height: 20px">已加载模型：{{ modelLoaded }}</div>
          </div>
        </el-aside>

        <!-- 可视化建模区的主要内容 -->
        <el-main @dragover.prevent ref="efContainerRef" id="efContainer "
          style="height: auto; width: 600px; padding: 0px;">
          <div
            style=" position: relative; height: 32%; font-size: 20px; color: #003e50; font-weight: 500; font-family:Arial, Helvetica, sans-serif;  background-position: center;">
            <div id="statusIndicator" class="status-indicator">未建立模型</div>

            <!-- <el-button type="primary" style="font-size: 18px; width: 180px;" @click="drawer = true">打开功能区</el-button> -->

            <DraggableContainer :reference-line-visible="false">
              <Vue3DraggableResizable :draggable="true" :resizable="false" v-for="(item, index) in nodeList" 
                :key="item.nodeId" class="node-info" :id="item.id" :style="item.nodeContainerStyle"
                :ref="el => nodeRef[index] = el" @click="resultShow(item)">
                <el-popover placement="bottom" title="参数配置" :width="400" trigger="contextmenu">
                  <!-- 调整参数 -->
                  <!-- 可视化建模区中的各节点所具有的参数与代码中menuList2中的参数是相对应的 -->
                  <el-row v-if="item.use_algorithm != null && item.id != '1.2' && item.id != '1.3' && item.id != '1.5'"
                    v-for="(value, key) in item.parameters[item.use_algorithm]"
                    :key="item.parameters[item.use_algorithm].keys" style="margin-bottom: 20px">
                    <el-col :span="8" style="align-content: center;"><span style="margin-left: 10px; font-size: 15px;">{{ labelsForParams[key] }}：</span></el-col>
                    <el-col :span="16">
                      <el-select v-model="item.parameters[item.use_algorithm][key]" collapse-tags collapse-tags-tooltip :teleported="false">
                        <el-option 
                          v-for="item in recommendParams[key]"
                          :key="item.value"
                          :label="item.label"
                          :value="item.value"
                          style="width: 200px; height: auto; background-color: white;" 
                        />
                      </el-select>
                    </el-col>
                  </el-row>
                  <!-- 特征提取选择要显示的特征 -->
                  <el-row v-if="item.id == '1.2'">
                    <el-col :span="8" style="align-content: center;"><el-text style="margin-left: 10px; font-size: 15px;">选择特征：</el-text></el-col>
                    <el-col :span="16">
                      <div class="m-4">
                        <el-select v-model="features" multiple collapse-tags collapse-tags-tooltip
                          placeholder="选择需要提取的特征" :teleported="false">
                          <el-option v-for="(value, key) in item.parameters[item.use_algorithm]" :label="key"
                            :value="key" style="width: 200px; background-color: white; padding: 0px;" />
                        </el-select>
                      </div>
                    </el-col>
                  </el-row>

                  <!-- 选择特征选择的规则以及设定规则的阈值 -->
                  <div v-if="item.id == '1.3'">
                    <el-radio-group v-model="item.parameters[item.use_algorithm]['rule']">
                      <el-radio :value="1" size="large">规则一</el-radio>
                      <el-radio :value="2" size="large">规则二</el-radio>
                    </el-radio-group>
                    <!-- 特征选择规则一 -->
                    <div v-if="item.parameters[item.use_algorithm]['rule'] == 1">
                      <div style="margin-top: 5px; margin-bottom: 15px;">
                        设定阈值后，将选择重要性系数大于该阈值的特征
                      </div>

                      <el-form>
                        <el-form-item label="阈值" >
                          <el-select v-model="item.parameters[item.use_algorithm]['threshold1']" size='medium' placeholder="请输入阈值" style="width: 250px;" :teleported="false">
                            <el-option 
                            v-for="item in recommendParams['threshold1'][item.use_algorithm]"
                            :key="item.value"
                            :label="item.label"
                            :value="item.value"
                            style="width: 200px; height: auto; background-color: white;" 
                            />
                          </el-select>
                        </el-form-item>
                      </el-form>
                    </div>
                    <!-- 特征选择规则二 -->
                    <div v-if="item.parameters[item.use_algorithm]['rule'] == 2">
                      <div style="margin-top: 5px; margin-bottom: 15px;">
                        设定阈值后，将根据特征的重要性，由高到低地选择特征，直到所选特征的重要性的总和占所有特征的重要性比例不小于该阈值，其中所有特征的重要性占比为1
                      </div>

                      <el-form>
                        <el-form-item label="阈值" >
                          <el-select v-model="item.parameters[item.use_algorithm]['threshold2']" size='medium' placeholder="请输入阈值" style="width: 250px;" :teleported="false">
                            <el-option 
                            v-for="item in recommendParams['threshold2'][item.use_algorithm]"
                            :key="item.value"
                            :label="item.label"
                            :value="item.value"
                            style="width: 200px; height: auto; background-color: white;" 
                            />
                          </el-select>
                        </el-form-item>
                      </el-form>
                    </div>
                  </div>
                  <!-- 无量纲化参数设置 -->
                  <div v-if="item.id == '1.5'">
                    <div>是否使用模型训练时的标准化方法</div>
                    <el-radio-group v-model="item.parameters[item.use_algorithm]['useLog']">
                      <el-radio :value="true" size="large">是</el-radio>
                      <el-radio :value="false" size="large">否</el-radio>
                    </el-radio-group>
                  </div>

                  <template #reference>
                    <div class="node-info-label el-dropdown-link font-style: italic;" :id=item.id>{{ item.label_display
                      }}
                      <div style="
                      position: absolute; left: 55px; top: 35px; width: 6px; height: 6px;
                      border: 2px solid #80a5ba; /* 边框颜色*/
                      border-radius: 50%; /* 设置为50%以创建圆形 */
                      background-color: transparent; /* 背景设置为透明，实现空心效果 */
                      /* 其他样式，如 cursor 可以设置拖拽时的鼠标光标形状 */
                      cursor: move; /* 鼠标悬停时显示可移动的光标 */" @mouseup="handleMouseup($event, item)"></div>
                      <el-button type="danger" icon="Delete" circle size="small" class="deleteButton"
                        @click="deleteNode(item.nodeId)" :disabled="modelSetup" />
                    </div>

                  </template>
                </el-popover>
                <!-- @contextmenu="params_setting(item[parameters])" -->
                <div class="node-drag" :id="item.id"></div>
              </Vue3DraggableResizable>
            </DraggableContainer>
            <!-- <div style="width: 1000px; height: 100px; background-color: #88b6fb;"></div> -->
            <div
              style="position: absolute; right: 250px; bottom: 10px; width: 600px; height: auto;display: flex; justify-content: space-between; align-items: center;">
         
              <el-space size="large">

                <el-button type="info" round style="width: 125px; font-size: 17px; background-color: #E6A23C;"
                  @click="handleClear" icon="Refresh">
                  清空模型
                </el-button>

                <el-button v-if="!toRectifyModel" type="primary" :disabled="canCompleteModeling"
                  @mouseover="CompleteModeling" round style="width: 125px; font-size: 17px;" @click="finishModeling"
                  icon="Check">
                  完成建模
                </el-button>
              
                <el-button v-if="toRectifyModel" type="primary" :disabled="canCompleteModeling"
                  @mouseover="CompleteModeling" round style="width: 125px; font-size: 17px;" @click="rectifyModel"
                  icon="Edit">
                  修改模型
                </el-button>

                
                <el-button type="primary" :disabled="canCheckModel" @mouseover="checkModeling" round
                  style="width: 125px; font-size: 17px; " @click="checkModel" icon="Search">
                  检查模型
                </el-button>

           
                <el-button type="primary" :disabled="canSaveModel" @mouseover="saveModeling" round
                  style="width: 125px; font-size: 17px;" @click="saveModelSetting(true, [])" icon="Finished">
                  保存模型
                </el-button>
              
                <el-button type="success" round style="width: 125px; font-size: 17px; " @click="run"
                  icon="SwitchButton" :disabled="canStartProcess || processIsShutdown" @mouseover="startModeling">
                  开始运行
                </el-button>
                <el-button :disabled="canShutdown" type="danger" round style="width: 125px; font-size: 17px;"
                  @click="shutDown" icon="Close">
                  终止运行
                </el-button>
              </el-space>

            </div>
          </div>

          <div class="resultsContainer" style="background-color: white;">

            <!-- 点击在可视化建模区展示算法的具体介绍 -->
            <el-scollbar height="400px">
              <a-button type="text" style="position: absolute; top: 5px; right: 5px" v-if="showPlainIntroduction" @click="showPlainIntroduction = false">关闭</a-button>
              <v-md-preview v-if="showPlainIntroduction" :text="introductionToShow"
              style="text-align: left;"></v-md-preview>
              <v-md-preview v-if="showStatusMessage" :text="statusMessageToShow"
                style="text-align: center;"></v-md-preview>
            </el-scollbar>
            
            <!-- 显示程序运行的进度条 -->
            <el-progress v-if="processing" :percentage="percentage" :indeterminate="true" />
            <!-- <iframe id='my_gradio_app' style="width: 1200px; height: 570px;" :src="refreshPardon" frameborder="0"
              v-if="isShow">
            </iframe> -->

            <!-- 显示结果 -->
            <el-scrollbar height="550px" v-if="canShowResults" style="background-color: white;">
              <!-- 健康评估可视化 -->
              <el-tabs class="demo-tabs" type="border-card" v-model="activeName1" v-if="displayHealthEvaluation">
                <el-tab-pane label="层级有效指标" name="first">
                  <img :src="healthEvaluationFigure1" alt="figure1" id="health_evaluation_figure_1"
                    class="result_image" style="width: auto; height: 450px;" />
                </el-tab-pane>
                <el-tab-pane label="指标权重" name="second">
                  <img :src="healthEvaluationFigure2" alt="figure2" id="health_evaluation_figure_2"
                    class="result_image" style="width: auto; height: 450px;" />
                </el-tab-pane>
                <el-tab-pane label="评估结果" name="third">
                  <el-col>
                    <img :src="healthEvaluationFigure3" alt="figure3" id="health_evaluation_figure_3"
                      class="result_image" style="width: auto; height: 360px;" />
                    <br>
                    <div style="width: 1000px; margin-left: 250px;">
                      <el-text :v-model="healthEvaluation" style="font-weight: bold; font-size: 18px;">{{
                        healthEvaluation
                      }}</el-text>
                    </div>
                  </el-col>
                </el-tab-pane>
              </el-tabs>
              <!-- 特征提取可视化 -->
              <el-table :data="transformedData" style="width: 96%; margin-top: 20px;"
                v-if="displayFeatureExtraction">
                <el-table-column v-for="column in columns" :key="column.prop" :prop="column.prop" :label="column.label"
                  :width="column.width">
                </el-table-column>
              </el-table>
              <!-- 特征选择可视化 -->
              <div v-if="displayFeatureSelection">
                <img :src="featureSelectionFigure" alt="feature_selection_figure1" class="result_image"
                  style="width: 900px; height: 450px;" />
                <br>
                <div style="width: 1000px; margin-left: 250px;">
                  <span style="font-weight: bold; font-size: 15px;">根据规则：{{ selectFeatureRule }}，选取特征结果为： {{
                    featuresSelected }}</span>
                </div>
              </div>
              <!-- 故障诊断可视化 -->
              <div v-if="displayFaultDiagnosis" style="margin-top: 20px; font-size: 18px;">
                <div style="width: 1000px; margin-left: 250px; font-weight: bold">
                  故障诊断结果为： 由输入的振动信号，根据故障诊断算法得知该部件<span :v-model="faultDiagnosis"
                    style="font-weight: bold; color: red;">{{
                      faultDiagnosis }}</span>
                </div>

                <br>
                <img :src="faultDiagnosisFigure" alt="fault_diagnosis_figure" class="result_image"
                  style="width: 800px; height: 450px;" />
              </div>
              <!-- 故障趋势预测可视化 -->
              <div v-if="displayFaultRegression" style="margin-top: 20px; font-size: 18px;">
                <div style="width: 1000px; margin-left: 250px;  font-weight: bold">
                  经故障诊断算法，目前该部件<span :v-model="faultRegression" style="font-weight: bold; color: red;">{{
                    faultRegression
                  }}</span>
                  <span v-if="!haveFault" :v-model="timeToFault" style="font-weight: bold;">根据故障预测算法预测，该部件{{
                    timeToFault
                  }}后会出现故障</span>
                </div>
                <br>
                <img :src="faultRegressionFigure" alt="fault_regression_figure" class="result_image"
                  style="width: 800px; height: 450px;" />
              </div>
              <!-- 插值处理结果可视化 -->
              <!-- <div v-if="displayInterpolation" style="margin-top: 20px; font-size: 18px;">
                <br>
                <img :src="interpolationFigure" alt="interpolation_figure" class="result_image"
                  style="width: 900px; height: 450px;" />
              </div> -->
              <el-tabs v-model="activeName3" v-if="displayInterpolation" type="border-card">
                <el-tab-pane v-for="item in interpolationResultsOfSensors" :key="item.name" :label="item.label" :name="item.name">
                  <img :src="interpolationFigures[item.name - 1]" alt="figure" id="figure"
                  class="result_image" style="width: 900px; height: 450px;" />
                </el-tab-pane>
              </el-tabs>
              <!-- 无量纲化可视化 -->
              <div v-if="displayNormalization" style="margin-top: 20px; font-size: 18px;">
                <div  v-if="normalizationResultType == 'table'">
                  <div style="font-size: large;">原数据</div>
                  <el-table :data="normalizationFormdataRaw" style="width: 96%; margin-top: 20px;"
                  >
                    <el-table-column v-for="column in normalizationColumns" :key="column.prop" :prop="column.prop" :label="column.label"
                      :width="column.width">
                    </el-table-column>
                  </el-table>
                  <br>
                  <div style="font-size: large;">标准化后数据</div>
                  <el-table :data="normalizationFormdataScaled" style="width: 96%; margin-top: 20px;"
                  >
                    <el-table-column v-for="column in normalizationColumns" :key="column.prop" :prop="column.prop" :label="column.label"
                      :width="column.width">
                    </el-table-column>
                  </el-table>
                </div>
                <el-tabs v-model="activeName4" v-if="normalizationResultType == 'figure'" type="border-card">
                  <el-tab-pane v-for="item in normalizationResultsSensors" :key="item.name" :label="item.label" :name="item.name">
                    <img :src="normalizationResultFigures[item.name - 1]" alt="figureOfSensor" id="figure"
                    class="result_image" style="width: 900px; height: 450px;" />
                  </el-tab-pane>
                </el-tabs>
                
              </div>
              <!-- 小波降噪可视化 -->
              
              <!-- <img :src="denoiseFigure" alt="denoise_figure" class="result_image"
                style="width: 900px; height: 450px;" /> -->
              <el-tabs v-model="activeName2" class="demo-tabs" v-if="displayDenoise" type="border-card">
                <el-tab-pane v-for="item in waveletResultsOfSensors" :key="item.name" :label="item.label" :name="item.name">
                  <img :src="denoiseFigures[item.name - 1]" alt="figure" id="figure"
                  class="result_image" style="width: 900px; height: 450px;" />
                </el-tab-pane>
              </el-tabs>
          
            </el-scrollbar>

          </div>

        </el-main>


        <!-- 以抽屉的形式打开功能区 -->

        <el-drawer v-model="modelsDrawer" direction="ltr">

          <div style="display: flex; flex-direction: column; ">
            <el-col>

              <h2 style=" margin-bottom: 25px; color: #253b45;">历史模型</h2>

              <el-table :data="fetchedModelsInfo" height="500" stripe style="width: 100%">
                <el-popover placement="bottom-start" title="模型信息" :width="400" trigger="hover">
                </el-popover>
                <el-table-column :width="100" property="id" label="序号" />
                <el-table-column :width="150" property="model_name" label="模型名称" show-overflow-tooltip/>
                <el-table-column :width="280" label="操作">
                  <template #default="scope">
                    <el-button size="small" type="primary" style="width: 50px;" @click="useModel(scope.row)">
                      使用
                    </el-button>
                    <el-button size="small" type="danger" style="width: 50px;"
                      @click="deleteModel(scope.$index, scope.row)">
                      删除
                    </el-button>
                    <el-popover placement="bottom" :width='500' trigger="click">
                      <el-descriptions :title="modelName" :column="3" :size="size" direction="vertical"
                      >
                        <el-descriptions-item label="使用模块" :span="3">
                          <el-tag size="small" v-for="algorithm in modelAlgorithms">{{ algorithm }}</el-tag>
                        </el-descriptions-item>
                        <el-descriptions-item label="算法参数" :span="3">
                          <div v-for="item in modelParams">{{ item.模块名 }}: {{ item.算法 }}</div>
                        </el-descriptions-item>
                      </el-descriptions>
                      <template #reference>
                        <el-button size="small" type="info" style="width: 80px" @click="showModelInfo(scope.row)">
                          查看模型
                        </el-button>
                      </template>
                    </el-popover>
                  </template>
                </el-table-column>
              </el-table>

              <el-dialog v-model="deleteModelConfirmVisible" title="提示" width="500">
                <span style="font-size: 20px;">确定删除该模型吗？</span>
                <template #footer>
                  <el-button
                    style="width: 150px"
                    @click="deleteModelConfirmVisible = false"
                    >取消</el-button
                  >
                  <el-button
                    style="width: 150px; margin-right: 70px"
                    type="primary"
                    @click="deleteModelConfirm"
                    >确定</el-button
                  >
                </template>
              </el-dialog>
            </el-col>
          </div>
        </el-drawer>

        <!-- 以抽屉的形式打开用户历史数据 -->
        <el-drawer v-model="dataDrawer" direction="ltr">
          <div style="display: flex; flex-direction: column">
            <el-col>
              <h2 style="margin-bottom: 25px; color: #253b45">用户数据文件</h2>

              <el-table :data="fetchedDataFiles" height="500" stripe style="width: 100%">
                <el-table-column :width="150" property="dataset_name" label="文件名称" show-overflow-tooltip/>
                <el-table-column :width="200" property="description" label="文件描述" show-overflow-tooltip/>
                <el-table-column :width="200" label="操作">
                  <template #default="scope">
                    <el-button
                      size="small"
                      type="primary"
                      style="width: 50px"
                      @click="useDataset(scope.row)"
                      :loading="loadingData"
                    >
                      使用
                    </el-button>
                    <el-button
                      size="small"
                      type="danger"
                      style="width: 50px"
                      @click="deleteDataset(scope.$index, scope.row)"
                    >
                      删除
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>

              <el-dialog
                v-model="deleteDatasetConfirmVisible"
                title="提示"
                width="500"
              >
                <span style="font-size: 20px">确定删除该数据文件吗？</span>
                <template #footer>
                  <el-button
                    style="width: 150px"
                    @click="deleteDatasetConfirmVisible = false"
                    >取消</el-button
                  >
                  <el-button
                    style="width: 150px; margin-right: 70px"
                    type="primary"
                    @click="deleteDatasetConfirm"
                    >确定</el-button
                  >
                </template>
              </el-dialog>
            </el-col>
           
          </div>

        </el-drawer>

      </el-container>

    </el-container>
    <el-dialog v-model="dialogModle" title="保存模型" draggable width="30%">
      <el-form :model="modelInfoForm">
        <el-form-item label="模型名称" :label-width='140'>
          <el-input style="width: 160px;" v-model="modelInfoForm.name" autocomplete="off" />
        </el-form-item>
      </el-form>
      <span class="dialog-footer">
        <el-button style="margin-left: 85px; width: 150px;" @click="dialogModle = false">取消</el-button>
        <el-button style="width: 150px;" type="primary" @click="saveModelConfirm">确定</el-button>
      </span>
    </el-dialog>
  </div>
</template>

<script lang="ts" setup>

import { onMounted, nextTick, ref, watch } from 'vue'
import { jsPlumb } from 'jsplumb'
import { ElNotification, ElMessage, ElMessageBox } from "element-plus";
import axios from 'axios';
import { DraggableContainer } from "@v3e/vue3-draggable-resizable";
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import uploadDatafile from './uploadDatafile.vue';
import api from '../utils/api.js'
import { labelsForAlgorithms, plainIntroduction, algorithmIntroduction, labelsForParams } from '../components/constant.ts'



const operationHelpDialogVisible = ref(false)  // 用户操作指南对话框
const userHelpDialogVisible = ref(false)  // 用户使用教程对话框
const userHelpDialogScrollbar = ref(null)

// 在使用教程中滚动到指定部分的方法  
const scrollTo = (sectionId: any) => {  
    if (userHelpDialogScrollbar.value) {  
      const element = userHelpDialogScrollbar.value.querySelector(`#${sectionId}`);  
      if (element) {  
        userHelpDialogScrollbar.value.scrollTop = element.offsetTop;  
      }  
    }  
  };  

const router = useRouter()  // 页面路由

const dialogVisible = ref(false)

const activeName = ref('first')    // 控制标签页

const modelsDrawer = ref(false)   // 控制模型列表的抽屉
const dataDrawer = ref(false)     // 控制数据文件的抽屉

//控制按钮失效变量
const canStartProcess = ref(true)

const canCompleteModeling = computed(() => {
  if (nodeList.value.length > 0 && !modelHasBeenSaved) {
    return false
  } else {
    return true
  }
})
const canCheckModel = ref(true)
const canSaveModel = ref(true)
const canShutdown = ref(true)

// 这是为了显示算法列表，以及完成算法参数定义等操作，所定义的数据结构
// 其中节点的id为算法的id，label为算法的名称，parameters为算法的参数，use_algorithm为当前该模块所使用的算法名称，tip_show为是否显示提示信息的标志，tip为提示信息
const menuList2 = ref([{
  label: '预处理算法', id: '1', options: [
    {
      label: '插值处理', id: '1.1', use_algorithm: null, parameters: {
        'neighboring_values_interpolation': {},
        'bicubic_interpolation': {},
        'lagrange_interpolation': {},
        'newton_interpolation': {},
        'linear_interpolation': {},
        'deeplearning_interpolation': {}
      }, tip_show: false, tip: '对输入信号进行插值'
    },
    {
      label: '特征提取', id: '1.2', use_algorithm: null, parameters: {
        'time_domain_features': { 均值: false, 方差: false, 标准差: false, 偏度: false, 峰度: false, 四阶累积量: false, 六阶累积量: false, 最大值: false, 最小值: false, 中位数: false, 峰峰值: false, 整流平均值: false, 均方根: false, 方根幅值: false, 波形因子: false, 峰值因子: false, 脉冲因子: false, 裕度因子: false },
        'frequency_domain_features': { 重心频率: false, 均方频率: false, 均方根频率: false, 频率方差: false, 频率标准差: false, 谱峭度的均值: false, 谱峭度的峰度: false, },
        'time_frequency_domain_features': {
          均值: false, 方差: false, 标准差: false, 峰度: false, 偏度: false, 四阶累积量: false, 六阶累积量: false, 最大值: false, 最小值: false, 中位数: false, 峰峰值: false, 整流平均值: false, 均方根: false, 方根幅值: false, 波形因子: false, 峰值因子: false, 脉冲因子: false, 裕度因子: false,
          重心频率: false, 均方频率: false, 均方根频率: false, 频率方差: false, 频率标准差: false, 谱峭度的均值: false, 谱峭度的峰度: false,
        },
        'time_domain_features_multiple': { 均值: false, 方差: false, 标准差: false, 偏度: false, 峰度: false, 四阶累积量: false, 六阶累积量: false, 最大值: false, 最小值: false, 中位数: false, 峰峰值: false, 整流平均值: false, 均方根: false, 方根幅值: false, 波形因子: false, 峰值因子: false, 脉冲因子: false, 裕度因子: false },
        'frequency_domain_features_multiple': { 重心频率: false, 均方频率: false, 均方根频率: false, 频率方差: false, 频率标准差: false, 谱峭度的均值: false, 谱峭度的峰度: false, },
        'time_frequency_domain_features_multiple': {
          均值: false, 方差: false, 标准差: false, 峰度: false, 偏度: false, 四阶累积量: false, 六阶累积量: false, 最大值: false, 最小值: false, 中位数: false, 峰峰值: false, 整流平均值: false, 均方根: false, 方根幅值: false, 波形因子: false, 峰值因子: false, 脉冲因子: false, 裕度因子: false,
          重心频率: false, 均方频率: false, 均方根频率: false, 频率方差: false, 频率标准差: false, 谱峭度的均值: false, 谱峭度的峰度: false
        }
      }, tip_show: false, tip: '手工提取输入信号的特征'
    },
    {
      label: '无量纲化', id: '1.5', use_algorithm: null, parameters: {
        'max_min': {useLog: false},
        'z-score': {useLog: false},
        'robust_scaler': {useLog: false},
        'max_abs_scaler': {useLog: false}
      }, tip_show: false, tip: '对输入数据进行无量纲化处理'
    },
    {
      label: '特征选择', id: '1.3', use_algorithm: null, parameters: {
        'feature_imp': {rule: 1, threshold1: null, threshold2: null},
        'mutual_information_importance': {rule: 1, threshold1: null, threshold2: null},
        'correlation_coefficient_importance': {rule: 1, threshold: 0.005},
        'feature_imp_multiple': {rule: 1, threshold: 0.005},
        'mutual_information_importance_multiple': {rule: 1, threshold: 0.005},
        'correlation_coefficient_importance_multiple': {rule: 1, threshold: 0.005}
      }, tip_show: false, tip: '对提取到的特征进行特征选择'
    },
    {
      label: '小波变换', id: '1.4', use_algorithm: null, parameters: {
        'wavelet_trans_denoise': {'wavelet': '',
        'wavelet_level': ''}
      }, tip_show: false, tip: '对输入信号进行小波变换'
    }
  ], tip_show: false, tip: '包含添加噪声、插值以及特征提取等'
},
{
  label: '故障预测算法', id: '2', options: [
    {
      label: '故障诊断', id: '2.1', use_algorithm: null, parameters: {
        'random_forest': {},
        'svc': {},
        'gru': {},
        'lstm': {},
        'random_forest_multiple': {},
        'svc_multiple': {},
        'gru_multiple': {},
        'lstm_multiple': {},
        'ulcnn': {},
        'ulcnn_multiple': {},
        'spectrumModel': {},
        'spectrumModel_multiple': {}
      }, tip_show: false, tip: '根据提取特征对输入信号作故障诊断'
    },
    {
      label: '趋势预测', id: '2.2', use_algorithm: null, parameters: {
        'linear_regression': {},
        'linear_regression_multiple': {}
      }, tip_show: false, tip: '根据提取的信号特征对输入信号进行故障预测'
    }]
},
{
  label: '健康评估算法', id: '3', options: [
    {
      label: '层次分析模糊综合评估', id: '3.1', use_algorithm: null, parameters: {
        'FAHP': {},
        'FAHP_multiple': {}
      }, tip_show: false, tip: '将模糊综合评价法和层次分析法相结合的评价方法'
    }]
},
  // {
  //   label: '语音处理', id: '2', options: [{ label: '音频分离', id: '2.1', use_algorithm: null, parameters: { 'conformer': { num_workers: 8, layers: 64 }, 'sepformer': { num_workers: 16, layers: 64 } }, tip_show: false, tip: '可对输入的一维音频信号进行噪声分离' },
  //   { label: '声纹识别', id: '2.2', use_algorithm: null, parameters: { 'conformer': {}, 'lightweight_cnn_conformer': {} }, tip_show: false, tip: '根据输入的说话人语音识别说话人' }]
  // },

]);


// 特征选择使用的规则
const featureSelectionRule = ref('')

// 该方法用于判断是否显示背景图片
const background_IMG = () => {
  if (nodeList.value.length == 0) {
    document.querySelector('.el-main').classList.add('has-background');

  }
  if (nodeList.value.length >= 1) {
    document.querySelector('.el-main').classList.remove('has-background');
    document.querySelector('.el-main').style.backgroundImage = ''

  }
}

// 算法参数的推荐值
const recommendParams = {
  'wavelet': [{value: 'db1', label: 'db1'}, {value: 'db2', label: 'db2'}, {value: 'sym1', label: 'sym1'}, {value: 'sym2', label: 'sym2'}, {value: 'coif1', label: 'coif1'}],
  'wavelet_level': [{value: 1, label: '1'}, {value: 2, label: '2'}, {value: 3, label: '3'}],
  'threshold1': {'feature_imp': [{value: 0.005, label: '0.005'}, {value: 0.01, label: '0.01'}, {value: 0.02, label: '0.02'}, {value: 0.03, label: '0.03'}, {value: 0.04, label: '0.04'}, {value: 0.05, label: 0.05}, {value: 0.1, label: 0.1}],
  'mutual_information_importance': [{value: 0.1, label: '0.1'}, {value: 0.2, label: '0.2'}, {value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'}, ], 
  'mutual_information_importance_multiple': [{value: 0.3, label: '0.3'}, {value: 0.35, label: '0.35'}, {value: 0.4, label: '0.4'}, {value: 0.45, label: '0.45'}, {value: 0.5, label: '0.5'}], 
  'feature_imp_multiple': [{value: 0.01, label: '0.01'}, {value: 0.03, label: '0.03'}, {value: 0.05, label: '0.05'}, {value: 0.06, label: '0.06'}],
  'correlation_coefficient_importance_multiple': [{value: 0.58, label: '0.58'}, {value: 0.6, label: '0.6'}, {value: 0.62, label: '0.62'}, {value: 0.64, label: '0.64'}],
  'correlation_coefficient_importance': [{value: 0.1, label: '0.1'}, {value: 0.2, label: '0.2'}, {value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'}, ]},
  
  'threshold2': {'feature_imp': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}],
'mutual_information_importance': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}],
'feature_imp_multiple': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}],
'mutual_information_importance_multiple': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}],
'correlation_coefficient_importance_multiple': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}],
'correlation_coefficient_importance': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}]},
  // 'thresholdImpSingle1': [{value: 0.005, label: '0.005'}, {value: 0.01, label: '0.01'}, {value: 0.02, label: '0.02'}, {value: 0.03, label: '0.03'}, {value: 0.04, label: '0.04'}, {value: 0.05, label: 0.05}, {value: 0.1, label: 0.1}],
  // 'thresholdImgSingle2': [{value: 0.3, label: '0.3'}, {value: 0.4, label: '0.4'}, {value: 0.5, label: '0.5'}, {value: 0.6, label: '0.6'}, {value: 0.7, label: '0.7'}, {value: 0.8, label: '0.8'}, {value: 1, label: 1}]
  'scaleUseLogs': [{value: true, label: '使用训练模型时对数据的标准化方法'}, {value: false, label: '不使用训练模型时对数据的标准化方法'}]
}


// 监听特征选择的规则对应的阈值的初始值，以适应性的调整阈值的初始值

// 用于显示算法介绍
const introductionToShow = ref('# 你好世界')  // 需要展示在可视化建模区的算法介绍
const showPlainIntroduction = ref(false)

// 点击标签页切换单传感器和多传感器算法
const handleClick = (tab, event) => {
  console.log(tab, event)
}

// 算法介绍，点击算法选择区内的具体算法，将其算法介绍展示在可视化建模区
const showIntroduction = (algorithm) => {
  resultsViewClear()
  showStatusMessage.value = false
  showPlainIntroduction.value = true
  introductionToShow.value = plainIntroduction[algorithm]

}

// 算法选择菜单下拉展示
const menuDetailsSecond = ref({

})

const menuDetailsThird = ref({

})

// 特征提取所选择的特征
// const features = ref([])
const features = ref(['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
  '波形因子', '峰值因子', '脉冲因子', '裕度因子', '重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的峰度'])

//双向链表用于存储调用的模块顺序
class ListNode {
  constructor(value) {
    this.value = value;
    this.next = null;
  }
}

class LinkedList {
  constructor() {
    this.head = null;
    this.tail = null;
  }

  // 添加新元素到链表尾部
  append(value) {
    const newNode = new ListNode(value);

    if (!this.head) {
      this.head = newNode;
      this.tail = newNode;
    } else {
      this.tail.next = newNode;
      this.tail = newNode;
    }
  }

  // 在链表的头部添加新节点
  insertAtHead(value) {
    const newNode = new ListNode(value);

    if (!this.head) {
      this.head = newNode;
      this.tail = newNode;
    } else {
      newNode.next = this.head;
      this.head = newNode;
    }
  }

  // 打印链表所有元素
  print() {
    let current = this.head;
    while (current) {
      console.log(current.value);
      current = current.next;
    }
  }
  get_all_nodes() {
    let current = this.head;
    let nodeList = []
    while (current) {
      nodeList.push(current.value)
      current = current.next;
    }
    return nodeList
  }
  length() {
    if (this.head) {
      let len = 1
      let p = this.head.next
      while (p) {
        p = p.next
        len += 1
      }
      return len
    }
    return 0
  }
  search(target_value) {
    if (this.head == null) {
      return false
    } else {
      let current = this.head
      while (current) {
        if (current.value == target_value) {
          return current
        }
        current = current.next
      }
      return false
    }
  }
  searchPre(target_value) {
    if (this.head == null){
      return false
    } else {
      let current = this.head
      while (current && current.next) {
        if (current.next.value == target_value) {
          return current
        }
        current = current.next
      }
      return false
    }
  }
}

const logout = () => {
  router.push('/')
}

// 标签与节点id的转换
const display_label_to_id = (display_label) => {
  nodeList.value.forEach(node => {
    if (node.display_label == display_label) {
      return node.id
    }
  })
}
// const label_to_id = (label) => {
//   let node_list = nodeList.value.slice()
//   node_list.forEach(node => {
//     console.log('node: ', node)
//     if (node.label == label){
//       console.log('node_id: ', node.id)
//       return node.id
//     }
//   })
// }
// 节点标签到节点标识id的转换
function labelToId(label) {
  let nodeList1 = nodeList.value.slice()
  let nodeIdToFind = 0
  nodeList1.forEach(node => {
    if (node.label == label) {
      nodeIdToFind = node.id
    }
  })
  return nodeIdToFind
}

// 建立模型的连线操作, 挂在到onMounted中
const linkedList = new LinkedList()
onMounted(() => {
  username.value = window.localStorage.getItem('username') || '用户名未设置'
  console.log('username: ', username.value)

  // 当进行建模的时候隐藏可视化建模区的背景文字
  document.querySelector('.el-main').classList.add('has-background');
  plumbIns = jsPlumb.getInstance()
  jsPlumbInit()

  plumbIns.bind("connection", function (info) {
    let sourceId = info.connection.sourceId
    let targetId = info.connection.targetId

    let id_to_label = {}

    nodeList.value.forEach(node => {
      let id = node.id
      let label = node.label
      id_to_label[id] = label
    })
    if (linkedList.head == null) {
      linkedList.append(id_to_label[sourceId])
      linkedList.append(id_to_label[targetId])
    } else {
      if (linkedList.head.value == id_to_label[targetId]) {
        linkedList.insertAtHead(id_to_label[sourceId])
      } else {
        linkedList.append(id_to_label[targetId])
      }
    }
    // 除去在linkedList中的节点，其他节点不能作为连线操作的出发点
    let linked = linkedList.get_all_nodes()
    // for(let [value, key] of id_to_label){
    //   if (linked.indexOf(key) == -1){
    //     plumbIns
    //   }
    // }
    console.log('linked: ' + linked)
  })

  // plumbIns.bind('beforeConnect', function (info) {
  //   let sourceId = info.connection.sourceId
  //   let targetId = info.connection.targetId
  //   if (sourceId == '3.1') {
  //     return false
  //   }
  // })
})

const deff = {
  jsplumbSetting: {
    // 动态锚点、位置自适应
    Anchors: ['Right', 'Left'],
    anchor: ['Right', 'Left'],
    // 容器ID
    Container: 'efContainer',
    // 连线的样式，直线或者曲线等，可选值:  StateMachine、Flowchart，Bezier、Straight
    // Connector: ['Bezier', {curviness: 100}],
    // Connector: ['Straight', { stub: 20, gap: 1 }],
    Connector: ['Flowchart', { stub: 30, gap: 1, alwaysRespectStubs: false, midpoint: 0.5, cornerRadius: 10 }],
    // Connector: ['StateMachine', {margin: 5, curviness: 10, proximityLimit: 80}],
    // 鼠标不能拖动删除线
    ConnectionsDetachable: false,
    // 删除线的时候节点不删除
    DeleteEndpointsOnDetach: false,
    /**
     * 连线的两端端点类型：圆形
     * radius: 圆的半径，越大圆越大
     */
    Endpoint: ['Dot', { radius: 10, cssClass: 'ef-dot', hoverClass: 'ef-dot-hover' }],

    EndpointStyle: { fill: '#1879ffa1', outlineWidth: 3, },
    // 是否打开jsPlumb的内部日志记录
    LogEnabled: true,
    /**
     * 连线的样式
     */
    PaintStyle: {
      // 线的颜色
      stroke: '#4CAF50',
      // 线的粗细，值越大线越粗
      strokeWidth: 7,
      // 设置外边线的颜色，默认设置透明
      outlineStroke: 'transparent',
      // 线外边的宽，值越大，线的点击范围越大
      outlineWidth: 5,
    },
    DragOptions: { cursor: 'pointer', zIndex: 2000 },
    ConnectionOverlays: [
      ['Custom', {
        create() {
          const el = document.createElement('div')
          // el.innerHTML = '<select id=\'myDropDown\'><option value=\'foo\'>foo</option><option value=\'bar\'>bar</option></select>'
          return el
        },
        location: 0.7,
        id: 'customOverlay',
      }],
    ],

    Overlays: [
      // 箭头叠加
      ['Arrow', {
        width: 25, // 箭头尾部的宽度
        length: 8, // 从箭头的尾部到头部的距离
        location: 1, // 位置，建议使用0～1之间
        direction: 1, // 方向，默认值为1（表示向前），可选-1（表示向后）
        foldback: 0.623, // 折回，也就是尾翼的角度，默认0.623，当为1时，为正三角
      }],
  
      ['Label', { label: '', location: 0.1, cssClass: 'aLabel', }],

    ],
    // 绘制图的模式 svg、canvas
    RenderMode: 'canvas',
    // 鼠标滑过线的样式
    HoverPaintStyle: { stroke: 'red', strokeWidth: 10 },
    // 滑过锚点效果
    EndpointHoverStyle: { fill: 'red' },
    Scope: 'jsPlumb_DefaultScope', // 范围，具有相同scope的点才可连接
  },
  /**
   * 连线参数
   */
  jsplumbConnectOptions: {
    isSource: true,
    isTarget: true,
    // 动态锚点、提供了4个方向 Continuous、AutoDefault
    // anchor: 'Continuous',
    // anchor: ['Continuous', { faces: ['left', 'right'] }],
    // 设置连线上面的label样式
    labelStyle: {
      cssClass: 'flowLabel',
    },
  },
  /**
   * 源点配置参数
   */
  jsplumbSourceOptions: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: '.node-drag',
    filterExclude: false,
    anchor: ['Continuous', { faces: ['right'] }],
    // 是否允许自己连接自己
    allowLoopback: false,
    maxConnections: -1,
  },

  jsplumbSourceOptions2: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: '.node-drag',
    filterExclude: false,
    // anchor: 'Continuous',
    // 是否允许自己连接自己
    allowLoopback: true,
    connector: ['Flowchart', { curviness: 50 }],
    connectorStyle: {
      // 线的颜色
      stroke: 'red',
      // 线的粗细，值越大线越粗
      strokeWidth: 1,
      // 设置外边线的颜色，默认设置透明
      outlineStroke: 'transparent',
      // 线外边的宽，值越大，线的点击范围越大
      outlineWidth: 10,
    },
    connectorHoverStyle: { stroke: 'red', strokeWidth: 2 },
  },
  jsplumbTargetOptions: {
    // 设置可以拖拽的类名，只要鼠标移动到该类名上的DOM，就可以拖拽连线
    filter: '.node-drag',
    filterExclude: false,
    // 是否允许自己连接自己
    anchor: ['Continuous', { faces: ['left'] }],
    allowLoopback: false,
    dropOptions: { hoverClass: 'ef-drop-hover' },
  },
}


const done = ref(false) // 控制模型可拖拽，当其值为true时不可拖拽模型
const dialogModle = ref(false)

let modelCheckRight = false  // 为真时表明通过模型检查

// 检查模型
const checkModel = () => {
  console.log(linkedList.get_all_nodes())
  let idToModule = {}
  let algorithms = []
  let algorithmSchedule = []
  let moduleSchedule = []
  // 如果只有一个模块则不需要建立流程
  if (nodeList.value.length == 1) {
    moduleSchedule.push(nodeList.value[0].label)
    algorithmSchedule.push(nodeList.value[0].label_display)
  } else {

    // 如果有多个模块则需要根据用户的连接动作去形成正确的模型流程
    let allConnections = plumbIns.getConnections();
    console.log('all_connections: ', allConnections)
    // 获取连线元素的单向映射
    let connectionsMap: any = {};  
    allConnections.forEach(connection => {  
      const sourceId = connection.sourceId; 
      const targetId = connection.targetId;  
    
      // 如果源元素ID不在connectionsMap中，则初始化为空数组  
      if (!connectionsMap[sourceId]) {  
          connectionsMap[sourceId] = [];  
      }
      connectionsMap[sourceId].push(targetId);  
    })
    
    // 寻找用户建立的模型流程逻辑上的第一个元素
    function findStartElement(connectionsMap: any) {  
      // 创建一个集合来存储所有元素的ID  
      const allElements = new Set(Object.keys(connectionsMap).concat(...Object.values(connectionsMap).map(list => list)));  
    
      // 遍历所有元素，查找没有入度的元素  
      for (const elementId of allElements) {  
          let hasIncomingConnection = false;  
          for (const connections of Object.values(connectionsMap)) {  
              if (connections.includes(elementId)) {  
                  hasIncomingConnection = true;  
                  break;  
              }  
          }  
          if (!hasIncomingConnection) {  
              return elementId;     // 找到没有入度的元素，即起点  
          }  
      }  
    
      // 如果没有找到没有入度的元素，则可能图不是线性的，或者connectionsMap构建有误  
      throw new Error("未找到起点元素.");  
    }
    let startElementId = findStartElement(connectionsMap);
    
    // 寻找逻辑上的下一个元素
    function findNextElementIdInSequence(currentElementId, connectionsMap) {  
      
      const connections = connectionsMap[currentElementId];  
    
      // 建立模型时，模型序列是线性的 
      if (connections && connections.length > 0) {  
          return connections[0]; // 返回序列中的下一个元素ID  
      }  
      return null;  
    }
    
    // 生成所建立模型的运行流程
    function traverseLinearSequence(startElementId, connectionsMap, visited = new Set(), order = []) {  
      // 检查是否已访问过当前元素  
      if (visited.has(startElementId)) {  
          return;  
      }  
    
      visited.add(startElementId); // 标记为已访问  
      order.push(startElementId); // 将元素添加到顺序数组中  
     
      let nextElementId = findNextElementIdInSequence(startElementId, connectionsMap);  
    
      if (nextElementId !== null) {  
          // 递归遍历下一个元素  
          traverseLinearSequence(nextElementId, connectionsMap, visited, order);  
      }  
      // 最终返回模型的运行流程
      return order;  
    }
    let sequenceOrder = traverseLinearSequence(startElementId, connectionsMap);  
    console.log('sequenceOrder: ', sequenceOrder);
    nodeList.value.forEach(node => {
      let id = node.id
      let label = node.label
      let algorithm = node.label_display
      idToModule[id] = label
      algorithms.push(algorithm)
    })
    
    sequenceOrder.forEach(id => {
      moduleSchedule.push(idToModule[id])
    });
    

    for (let i = 0; i < moduleSchedule.length; i++) {
      let module = moduleSchedule[i]
      nodeList.value.forEach(node => {
        if (node.label == module) {
          algorithmSchedule.push(node.label_display)
        }
      });
    }
  }
  

  let moduleStr = Object.values(moduleSchedule).join('')   // 所有模块的名称按顺序拼接起来的字符串
  let algorithmStr = Object.values(algorithmSchedule).join('')  // 所有模块中的算法名称按顺序拼接起来的字符串

  // 判断子串后是否有更多的文本
  const moreText = (text, substring) => {
    const position = text.indexOf(substring);
    if (position === -1) {
      return false;
    }
    const endPosition = position + substring.length;
    return endPosition < text.length;
  }

  // 判断一个子串后是否有另一个子串，其中subStrs2为包含需要寻找的子串的数组
  const checkSubstrings = (str, subStr1, subStrs2) => {
    const index1 = str.indexOf(subStr1);
    if (index1 !== -1) {
      // 如果 subStr1 存在  
      for (const subStr2 of subStrs2) {
        const index2 = str.indexOf(subStr2, index1 + subStr1.length);
        if (index2 !== -1) {
          // 如果在 subStr1 之后找到了 subStr2 中的任何一个则返回true  
          return true;
        }
      }
    }
    return false;
  }
  if (nodeList.value.length) {
    // 首先判断模型中是否存在1个以上的模块，如果模型中只有一个模块，判断其是否可以独立地运行而不需要其他模块的支持
    if (nodeList.value.length == 1) {
      if (!moduleStr.match('插值处理') && !moduleStr.match('特征提取') && !algorithmStr.match('GRU的故障诊断') && !algorithmStr.match('LSTM的故障诊断') && !algorithmStr.match('小波变换降噪')
    && !algorithmStr.match('一维卷积深度学习模型的故障诊断') && !algorithmStr.match('基于时频图的深度学习模型的故障诊断') && !moduleStr.match('无量纲化')) {
        let tip
        if (moduleStr.match('故障诊断')) {
          tip = '模型中包含故障诊断，建议在此之前进行特征提取和特征选择等操作'
        }
        else if (moduleStr.match('层次分析模糊综合评估')) {
          tip = '模型中包含层次分析模糊综合评估，建议在此之前进行特征提取和特征选择等操作'
        }
        else if (moduleStr.match('特征选择')) {
          tip = '模型中包含特征选择，建议在此之前进行特征提取等操作'
        }
        else if (moduleStr.match('趋势预测')) {
          tip = '模型中包含趋势预测，建议在此之前进行故障诊断'
        }
      
        ElMessage({
          message: '该算法无法单独使用，请结合相应的算法,' + tip,
          type: 'warning',
          showClose: true
        })
        // ElMessageBox.alert(
        //   '该算法无法单独使用，请结合相应的算法', '提示',
        //   {
        //     confirmButtonText: '确定',
        //     draggable: true,
        //     buttonSize: 'medium',
        //   }
        // )
        // 检查单个模块是否可以单独运行，或者是否需要一些必要的其他模块的支持
       
        return
      } else {
        // 无量纲化要检查是否使用模型训练师使用的标准化方法，对输入的原始信号无法使用模型训练时使用的标准化方法进行无量纲化
        if (moduleStr.match('无量纲化')){
          let node
          for (let item of nodeList.value){
            if (item.label == '无量纲化'){
              node = item
              break
            }
          }
          if (node?.parameters[node.use_algorithm]['useLog'] == true){
            // ElMessage({
            //   message: '如果要使用模型训练是使用的标准化方法进行无量纲化，请确保无量纲化模块之前对数据进行了特征提取，或者选择不使用模型训练时使用的标准化方法',
            //   type: 'warning',
            //   showClose: true
            // })
            ElMessageBox.alert('如果要使用模型训练时使用的标准化方法进行无量纲化，请确保无量纲化模块之前对数据进行了特征提取，或者在参数设置中选择不使用模型训练时使用的标准化方法', '提示', {
              confirmButtonText: '确定',
              draggable: true,
              buttonSize: 'medium',
            })
            
            return
          }
        }
        // 如果模块可以单独运行，再进行模型中各个模块的参数设置的检查
        let checkParamsRight = checkModelParams()
        if (checkParamsRight) {
          ElMessage({
            showClose: true,
            message: '模型正常，可以保存并运行',
            type: 'success'
          })
          modelCheckRight = true
          updateStatus('模型建立并已通过模型检查')
        } else {
          ElMessage({
            showClose: true,
            message: '请确保所有具有参数的模块的参数设置正确',
            type: 'warning'
          })
          return
        }
      }
    } else {
      // 检查模型中是否存在未被连接的模块
      if (linkedList.length() != nodeList.value.length) {
        ElMessage({
          message: '请确保图中所有模块均已建立连接，且没有多余的模块',
          type: 'warning'
        })
        return
      } else {
        // 模型正常连接的情况下进行模型逻辑以及模型参数的检查
        if (algorithmStr.match('多传感器') && algorithmStr.match('单传感器')) {
          ElMessage({
            showClose: true,
            message: '多传感器和单传感器的算法不能混合使用！',
            type: 'warning'
          })
          return
        }
        // if (moduleStr.match('特征选择故障诊断') && !moduleStr.match('特征提取特征选择故障诊断') && !moduleStr.match('特征提取特征选择无量纲化故障诊断')
        //     && !moduleStr.match('特征提取无量纲化特征选择故障诊断') && !algorithmStr.match('深度学习模型的故障诊断') && !algorithmStr.match('GRU的故障诊断') && !algorithmStr.match('LSTM的故障诊断')) {
        //   ElMessage({
        //     showClose: true,
        //     message: '因模型中包含故障诊断，建议在特征选择之前包含特征提取',
        //     type: 'warning'
        //   })
        //   return
        // } 
        if(moduleStr.match('故障诊断')) {
          // 如果是深度学习模型的故障诊断
          if (algorithmStr.match('深度学习模型的故障诊断') || algorithmStr.match('GRU的故障诊断') || algorithmStr.match('LSTM的故障诊断')){
            if (moduleStr.indexOf('故障诊断') > 0){
              // 检查深度学习模型的故障诊断之前是否包含不必要的模块
              let preModuleText = moduleStr.substring(0, moduleStr.indexOf('故障诊断'))
              if (preModuleText.match('特征提取') || preModuleText.match('特征选择') || preModuleText.match('无量纲化') || preModuleText.match('趋势预测')) {
                ElMessage({
                  message: '深度学习模型的故障诊断不需要人工提取特征，因此其之前不需要包含如特征提取、特征选择、无量纲化、趋势预测等不必要的模块！',
                  type: 'warning',
                  showClose: true
                })
                return
              }
            }
            // 如果使用深度学习模型的故障诊断之后有其他的模块
            if (moreText(moduleStr, '故障诊断')){
              let nextModuleText = moduleStr.substring(moduleStr.indexOf('故障诊断'), moduleStr.length)
              if (nextModuleText.match('趋势预测') || nextModuleText.match('层次分析模糊综合评估')) {
                // 如果同时包含趋势预测以及层次分析模糊综合评估
                let current: String
                if (nextModuleText.match('趋势预测') && nextModuleText.match('层次分析模糊综合评估')){
                  if (nextModuleText.indexOf('趋势预测') > nextModuleText.indexOf('层次分析模糊综合评估')){
                    ElMessage({
                      message: '注意趋势预测应该在层次分析模糊综合评估之前运行',
                      type: 'warning',
                      showClose: true
                    })
                    return 
                  }
                  else {
                    current = '趋势预测'
                  }
                }
                
                // 因为之前的深度学习模型的故障诊断无法为趋势预测或是健康评估提供样本特征，因此需要进行特征提取和特征选择
                if (nextModuleText.indexOf('趋势预测') == -1){
                  current = '层次分析模糊综合评估'
                }
                if (nextModuleText.indexOf('层次分析模糊综合评估') == -1){
                  current = '趋势预测'
                }
                // if (nextModuleText.indexOf('趋势预测') > nextModuleText.indexOf('层次分析模糊综合评估')){
                //   current = '层次分析模糊综合评估'
                // }else{
                //   current = '趋势预测'
                // }
                let preModuleText = nextModuleText.substring(0, nextModuleText.indexOf(current))
                if (!preModuleText.match('特征提取特征选择') && !preModuleText.match('特征提取无量纲化特征选择') && !preModuleText.match('特征提取特征选择无量纲化')){
                  ElMessage({
                    message: '建议在深度学习模型的故障诊断之后包含特征提取和特征选择模块',
                    type: 'warning',
                    showClose: true
                  })
                  return
                }
           
              }
            }
          }
          else {
            // 如果是传统机器学习的故障诊断
            let preModuleText = moduleStr.substring(0, moduleStr.indexOf('故障诊断'))
            // if (!preModuleText.match('特征提取') && !preModuleText.match('特征选择') && !preModuleText.match('特征提取无量纲化特征选择')){
            //   ElMessage({
            //     message: '因模型中包含机器学习的故障诊断，建议在故障诊断之前包含特征提取及特征选择',
            //     type: 'warning'
            //   })
            //   return
            // }
            // if (!preModuleText.match('特征提取特征选择') && !preModuleText.match('特征提取无量纲化特征选择') && !preModuleText.match('特征提取特征选择无量纲化')){
              
            //   ElMessage({
            //     message: '建议在特征提取之后进行特征选择',
            //     type: 'warning',
            //     showClose: true
            //   })
            //   return 
            // }
            // 机器学习的故障诊断之前不包含特征提取和特征选择
            if (!preModuleText.match('特征提取') && !preModuleText.match('特征选择')){
              ElMessage({
                message: '建议在故障诊断之前进行特征提取和特征选择',
                type: 'warning',
                showClose: true
              })
              return
            }
            else {
              // 如果特征提取和特征选择同时存在
              if (preModuleText.match('特征提取') && preModuleText.match('特征选择')){
                // 特征提取在特征选择之后，此时逻辑错误
                if (preModuleText.indexOf('特征提取') > preModuleText.indexOf('特征选择')){
                  ElMessage({
                    message: '建议在特征提取之后进行特征选择',
                    type: 'warning',
                    showClose: true
                  })
                  return
                }
              }
              else {
                // 只包含特征选择
                if (preModuleText.match('特征选择')) {
                  ElMessage({
                    message: '建议在特征提取之后再进行特征选择',
                    type: 'warning',
                    showClose: true
                  })
                  return
                }
                // 只包含特征提取
                if (preModuleText.match('特征提取')) {
                  ElMessage({
                    message: '因模型中包含机器学习的故障诊断，建议在特征提取之后进行特征选择',
                    type: 'warning',
                    showClose: true
                  })
                  return
                }
              }
            }
          }
        }
          // else {
          //   // 如果是机器学习的故障诊断
          //   if (moduleStr.indexOf('故障诊断') > 0) {
          //     let preModuleText = moduleStr.substring(0, moduleStr.indexOf('故障诊断'))
          //     if (!preModuleText.match('特征提取') && !preModuleText.match('特征选择') && !preModuleText.match('特征提取无量纲化特征选择')){
          //       ElMessage({
          //         message: '因模型中包含故障诊断，建议在故障诊断之前包含特征提取及特征选择',
          //         type: 'warning'
          //       })
          //       return
          //     }
          //     if (!preModuleText.match('特征提取特征选择') && !preModuleText.match('特征提取无量纲化特征选择') && !preModuleText.match('特征提取特征选择无量纲化')){
                
          //       ElMessage({
          //         message: '建议在特征提取之后进行特征选择',
          //         type: 'warning',
          //         showClose: true
          //       })
          //       return 
          //     }
          //   }
          // }
        if (moduleStr.match('特征提取故障诊断')) {
          let sourceId = labelToId('特征提取')
          let current = linkedList.search('特征提取')
          let next = current.next.value
          let targetId = labelToId(next)

          let connection = plumbIns.getConnections({ source: sourceId, traget: targetId })
          console.log('connection: ', connection)

          plumbIns.select({ source: sourceId, target: targetId }).setPaintStyle({
            stroke: '#E53935',
            strokeWidth: 7,
            outlineStroke: 'transparent',
            outlineWidth: 5,

          });
          ElMessage({
            showClose: true,
            message: '因模型中包含故障诊断，建议在特征提取之后包含特征选择',
            type: 'warning'
          })
          return
        } 
        if (moduleStr.match('层次分析模糊综合评估') && !moduleStr.match('特征提取')) {

          ElMessage({
            showClose: true,
            message: '因模型中包含层次分析模糊综合评估，建议在此之前包含特征提取',
            type: 'warning'
          })
          
          let current = linkedList.searchPre('层次分析模糊综合评估') // 寻找健康评估之前的节点，即不符合规则的节点
          // 红色表明报错连线
          
          let sourceId = labelToId(current.value)
          let targetId = labelToId('层次分析模糊综合评估')
          
          let connection = plumbIns.getConnections({ source: sourceId, traget: targetId })
          console.log('connection: ', connection)

          plumbIns.select({ source: sourceId, target: targetId }).setPaintStyle({
            stroke: '#E53935',
            strokeWidth: 7,
            outlineStroke: 'transparent',
            outlineWidth: 5,

          });
          return
        } 
        if (algorithmStr.match('深度学习模型的故障诊断') || algorithmStr.match('GRU的故障诊断') || algorithmStr.match('LSTM的故障诊断')) {
          // 如果使用深度学习的故障诊断之前有其他模块，则要进行限定
           
          if (moduleStr.indexOf('故障诊断') != 0){
            let preText = moduleStr.substring(0, moduleStr.indexOf('故障诊断')) 
            // 检查使用深度学习的故障诊断之前是否有特征提取等不合理的模块
            if (preText.match('特征提取') || preText.match('特征选择') || preText.match('无量纲化')) {
              ElMessage({
                showClose: true,
                message: '使用深度学习模型的故障诊断不需要进行特征提取或是特征选择，请删除相关模块！',
                type: 'warning'
              })
            }
            return
          }
        }
        // if (moduleStr.match('层次分析模糊综合评估') && (moduleStr.match('LSTM的故障诊断') || moduleStr.match('GRU的故障诊断'))) {
        //   ElMessage({
        //     showClose: true,
        //     message: '使用深度学习模型的故障诊断无法为健康评估提供有效的评估依据，建议使用机器学习的故障诊断配合健康评估！',
        //     type: 'warning'
        //   })
        //   return
        // }
        // 健康评估之后无法连接其他模块
        if (moduleStr.match('层次分析模糊综合评估') && moreText(moduleStr, '层次分析模糊综合评估')) {
          let sourceId = labelToId('层次分析模糊综合评估')
          let current = linkedList.search('层次分析模糊综合评估')
          let next = current.next.value
          let targetId = labelToId(next)

          plumbIns.select({ source: sourceId, target: targetId }).setPaintStyle({
            stroke: '#E53935',
            strokeWidth: 7,
            outlineStroke: 'transparent',
            outlineWidth: 5,

          });
          ElMessage({
            showClose: true,
            message: '注意健康评估之后无法连接更多的模块',
            type: 'warning'
          })
          return
        }
        if (algorithmStr.match('多传感器') && algorithmStr.match('单传感器')) {
          ElMessage({
            showClose: true,
            message: '针对单传感器的算法无法与针对多传感器的算法共用',
            type: 'warning'
          })
          return
        }
        if(moduleStr.match('无量纲化')) {
          console.log('算法中包含无量纲化')
          let node
          for(let item of nodeList.value){
            if (item.label.match('无量纲化')){
              node = item
              break
            }
          }
          let useLog = node.parameters[node.use_algorithm]['useLog']  // 获取无量纲化模块的参数
          if (moduleStr.indexOf('无量纲化') == 0){
            // 无量纲化处理前没有其他模块
            
            // 检查无量纲化参数设置是否合理
            if (useLog == true) {
             
              ElMessageBox.alert('如果要使用模型训练时使用的标准化方法进行无量纲化，请确保无量纲化模块之前对数据进行了特征提取，或者在参数设置中选择不使用模型训练时使用的标准化方法', '提示', {
                confirmButtonText: '确定',
                draggable: true,
                buttonSize: 'medium',
              })
              return
            }
          }
          else{
            
            // 检查无量纲化处理前的其他模块
            let preModule = moduleStr.substring(0, moduleStr.indexOf('无量纲化')) 
            if(preModule.match('特征提取') && useLog == false){
              ElMessageBox.alert(
                '因为无量纲化模块之前已经进行了特征提取，请在无量纲化的参数设置中选择使用模型训练时使用的标准化方法进行无量纲化',
                '提示',
                {
                  confirmButtonText: '确定',
                  draggable: true,
                  buttonSize: 'medium',
                }
              )
              return
            }
            else if (!preModule.match('特征提取') && useLog == true){
              ElMessageBox.alert(
                '无量纲化模块之前未进行特征提取，因此无法使用模型训练时使用的标准化方法进行无量纲化，请在无量纲化的参数设置中选择不使用模型训练时使用的标准化方法进行无量纲化',
                '提示',
                {
                  confirmButtonText: '确定',
                  draggable: true,
                  buttonSize: 'medium',
                }
              )
            }
          }

        }
        // if ((algorithmStr.match('LSTM的故障诊断') || algorithmStr.match('GRU的故障诊断')) && (checkSubstrings(moduleStr, '故障诊断', ['趋势预测', '健康评估']))){
        //   let nextModuleText = moduleStr.substring(moduleStr.indexOf('故障诊断'), moduleStr.length)
        //   if (!nextModuleText.match('特征提取') && !nextModuleText.match('特征选择')){
        //     ElMessage({
        //       message: '注意深度学习的算法并不能为线性回归的趋势预测或是健康评估提供需要的特征。',
        //       type: 'warning',
        //       showClose: true
        //     })
        //     return
        //   }
        //   // ElMessage({
        //   //   message: '注意深度学习的算法并不能为线性回归的趋势预测或是健康评估提供需要的特征。',
        //   //   type: 'warning',
        //   //   showClose: true
        //   // })
          
        //   let sourceId = labelToId('故障诊断')
        //   let current = linkedList.search('故障诊断')
        //   // 红色表明报错连线
        //   let next = current.next.value       // 寻找目标连线的源节点和目标节点
        //   let targetId = labelToId(next)
          
        //   // let connection = plumbIns.getConnections({ source: sourceId, traget: targetId })
        //   // console.log('connection: ', connection)

        //   // 通过jsPlumb实例对象的select方法选择连线，并设置连线的样式
        //   plumbIns.select({ source: sourceId, target: targetId }).setPaintStyle({
        //     stroke: '#E53935',
        //     strokeWidth: 7,
        //     outlineStroke: 'transparent',
        //     outlineWidth: 5,

        //   });
        //   return
        // }
        if (moduleStr.match('故障诊断')) {
          if (moreText(moduleStr, '故障诊断')) {
            if (!checkSubstrings(moduleStr, '故障诊断', ['层次分析模糊综合评估', '趋势预测'])) {
              ElMessage({
                showClose: true,
                message: '注意故障诊断之后仅能进行趋势预测或是健康评估！',
                type: 'warning'
              })
              // 将报错的连线标注为红色
              let sourceId = labelToId('故障诊断')
              let current = linkedList.search('故障诊断')
              let next = current.next.value       // 寻找目标连线的源节点和目标节点
              let targetId = labelToId(next)
              
              // let connection = plumbIns.getConnections({ source: sourceId, traget: targetId })
              // console.log('connection: ', connection)
              // 通过jsPlumb实例对象的select方法选择连线，并设置连线的样式
              plumbIns.select({ source: sourceId, target: targetId }).setPaintStyle({
                stroke: '#E53935',
                strokeWidth: 7,
                outlineStroke: 'transparent',
                outlineWidth: 5,

              });
              
              return
            }
          }
          if (algorithmStr.match('SVM的故障诊断')) {
            if (!moduleStr.match('无量纲化') || !checkSubstrings(moduleStr, '无量纲化', ['故障诊断'])) {
              ElMessage({
                showClose: true,
                message: '因模型中包含SVM的故障诊断，需要在此之前加入标准化操作',
                type: 'warning'
              })
              // 将报错的连线标注为红色
              let sourceId = labelToId('特征选择')
              let current = linkedList.search('特征选择')
              let next = current.next.value       // 寻找目标连线的源节点和目标节点
              let targetId = labelToId(next)
              
              // let connection = plumbIns.getConnections({ source: sourceId, traget: targetId })
              // console.log('connection: ', connection)
              // 通过jsPlumb实例对象的select方法选择连线，并设置连线的样式
              plumbIns.select({ source: sourceId, target: targetId }).setPaintStyle({
                stroke: '#E53935',
                strokeWidth: 7,
                outlineStroke: 'transparent',
                outlineWidth: 5,

              });
              return
            }
          }
        }
        // 进行模型参数设置的检查
        let check_params_right = checkModelParams()
        if (check_params_right) {
          ElMessage({
            showClose: true,
            message: '模型正常，可以保存并运行',
            type: 'success'
          })
          modelCheckRight = true
          updateStatus('模型建立并已通过模型检查')
        } else {
          ElMessage({
            showClose: true,
            message: '请确保所有具有参数的模块的参数设置正确',
            type: 'warning'
          })
          return
        }
      }
    }
  } else {
    ElMessage({
      message: '请先建立模型',
      type: 'warning'
    })
    return
  }
  canSaveModel.value = false
  // canStartProcess.value = false
}

// 进度条完成度
let processing = ref(false)
let percentage = ref(0)
// let timerId = null
let fastTimerId = null; // 快速定时器ID  
let slowTimerId = null; // 慢速定时器ID  

let responseResults = {}  // 从后端接收到的模型运行的结果数据

const username = ref('')  // 显示在用户界面中的用户名

// 创建一个CancelToken源  
let cancel;  
  
const source = axios.CancelToken.source();  
cancel = source.cancel; // 暴露cancel函数  


//上传文件后，点击开始运行以运行程序
const run = () => {

  if (!usingDatafile.value){
    ElMessage({
      message: '请先加载数据',
      type: 'warning'
    })
    return
  }

  const data = new FormData()
  console.log('datafile: ', usingDatafile.value)
  data.append("file_name", usingDatafile.value)
  data.append('params', JSON.stringify(contentJson))
  ElNotification.info({
    title: 'Waiting',
    message: '正在运行，请等待...'
  })
  canShutdown.value = false

  percentage.value = 0; // 重置进度条  

  fastTimerId = setInterval(() => {
    if (percentage.value < 50) {
      percentage.value += 10;
    } else {
      // 达到50%时，清除快速定时器并启动慢速定时器  
      clearInterval(fastTimerId);
      slowTimerId = setInterval(() => {
        if (percentage.value < 90) {
          percentage.value += 10;
        } else {
          // 达到100%时清除慢速定时器  
          clearInterval(slowTimerId);
        }
      }, 3000); // 每三秒更新一次  
    }
  }, 1000); // 每秒更新一次（在进度小于50%时）  

  // 显示进度条
  resultsViewClear()
  processing.value = true
  showStatusMessage.value = false
  showPlainIntroduction.value = false

  api.post('user/run_with_datafile_on_cloud/', data,
    {
      headers: { "Content-Type": 'multipart/form-data' },
      cancelToken: source.token, // 将cancelToken传递给axios  
    }
  ).then((response) => {
    console.log('response: ', response)
    console.log('response.status: ', response.status)
    if (response.status === 200) {
      
      if (!processIsShutdown.value) {
        ElNotification.success({
          title: 'Success',
          message: '程序运行完成',
        })
        responseResults = response.data.results
        // console.log('resoonse_results: ', response_results)
        missionComplete.value = true
        // setTimeout(function () { processing.value = false; percentage.value = 50 }, 500)
        // percentage.value = 100
        // clearInterval(timerId);
        clearInterval(fastTimerId);
        clearInterval(slowTimerId);
        setTimeout(function () { processing.value = false }, 700)
        percentage.value = 100;
        canShutdown.value = true
        statusMessageToShow.value = statusMessage.success
        resultsViewClear()

        showStatusMessage.value = true
        showPlainIntroduction.value = false
      } else {
        processIsShutdown.value = false
      }
    }
    
    // else if (response.status === 500) {
    //   ElNotification.error({
    //     title: 'ERROR',
    //     message: '处理失败，请重试',
    //   })
    //   loading.value = false
    // }
  })
    .catch(error => {
    
      if (error.response) {
        // 请求已发出，服务器响应了状态码，但不在2xx范围内  
        console.log(error.response.status); // HTTP状态码  
        console.log(error.response.statusText); // 状态消息  

      } else if (error.request) {
        // 请求已发起，但没有收到响应  
        console.log(error.request);
      } else {
        // 设置请求时触发了错误  
        console.error('Error', error.message);
      }

      ElNotification.error({
        title: 'Error',
        message: error.response.data.message,
      })
      loading.value = false
      processing.value = false

      canShutdown.value = true
      statusMessageToShow.value = statusMessage.error
      resultsViewClear()
      showStatusMessage.value = true
      missionComplete.value = false

    })
}

// 用于判断该程序是否是正常运行结束的，如果该变量为真，表示为手动终止运行
const processIsShutdown = ref(false)  


// 终止模型的运行
const shutDown = () => {
  api.get('/shut'  ).then((response: any) => {
    if (response.data.status == 'shutdown' && processing.value == true) {
      loading.value = false
      processing.value = false
      missionComplete.value = false
      ElNotification.info({
        title: 'INFO',
        message: '进程已终止'
      })
      resultsViewClear()
      processIsShutdown.value = true
      statusMessageToShow.value = statusMessage.shutdown
      showStatusMessage.value = true
      canShutdown.value = true
      // canStartProcess.value = false
      // cancel('Operation canceled by the user.');  
    }
  }).catch(function (error: any) {
    // 处理错误情况  
    ElNotification.error({
      title: 'ERROR',
      message: '请求终端进程失败'
    })
    console.log('请求中断进程失败：' + error)
  });
}


const isShow = ref(false)
const selects = ref(false)

const efContainerRef = ref()
const nodeRef = ref([])

const nodeList = ref([])   // 保存可视化建模区中的各节点的列表

// 前端向后端传递的要运行的模型的信息，由包括的模块、模块使用的算法、使用的参数、模块的运行顺序组成
const contentJson = {
  'modules': [],
  'algorithms': {},
  'parameters': {},
  'schedule': []
}

let plumbIns   // 实例化的jsPlumb对象，实现用户建模的连线操作
let missionComplete = ref(false)
let loading = ref(false)
let modelSetup = ref(false)

// 清除页面中的内容，包括使用的模型、文件和算法介绍等信息
const handleClear = () => {
  done.value = false
  nodeList.value = []  // 可视化建模区的节点列表
  // features.value = []  // 特征提取选择的特征
  features.value = ['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
    '波形因子', '峰值因子', '脉冲因子', '裕度因子', '重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的标准差', '谱峭度的峰度', '谱峭度的偏度']
  jsonClear()    // 向后端发送的模型信息
  isShow.value = false
  plumbIns.deleteEveryConnection()
  plumbIns.deleteEveryEndpoint()
  linkedList.head = null
  linkedList.tail = null
  missionComplete.value = false // 程序处理完成
  modelSetup.value = false   // 模型设置完成
  showPlainIntroduction.value = false
  showStatusMessage.value = false
  modelHasBeenSaved = false  //复用历史模型，不做模型检查
  toRectifyModel.value = false  // 禁用修改模型
  canCompleteModeling.value = true
  canCheckModel.value = true
  canSaveModel.value = true
  processIsShutdown.value = false
  canStartProcess.value = true
  modelLoaded.value = '无'

  updateStatus('未建立模型')

  resultsViewClear()
}

// 用于清空向后端传递的要运行的模型的信息
const jsonClear = () => {
  contentJson.modules = []
  contentJson.algorithms = {}
  contentJson.parameters = {}
  contentJson.schedule = []
}
const jsPlumbInit = () => {
  plumbIns.importDefaults(deff.jsplumbSetting)
}

//处理拖拽，初始化节点的可连接状态
const handleDragend = (ev, algorithm, node) => {

  // 拖拽进来相对于地址栏偏移量
  const evClientX = ev.clientX
  const evClientY = ev.clientY
  let left
  if (evClientX < 300){
    left = evClientX + 'px'
  }
  else{
    left = evClientX - 300 + 'px'
  }
  
  let top = 50 + 'px'
  const nodeId = node.id
  const nodeInfo = { 
    label_display: labelsForAlgorithms[algorithm],   // 具体算法的名称
    label: node.label,      // 算法模块名称
    id: node.id,
    nodeId,
    nodeContainerStyle: {
      left: left,
      top: top,
    },
    use_algorithm: algorithm,
    parameters: node.parameters
  }

  if (nodeInfo.label_display.indexOf('时域和频域') > -1) {
    features.value = ['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
      '波形因子', '峰值因子', '脉冲因子', '裕度因子', '重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的峰度']
  }else {
    if (nodeInfo.label_display.indexOf('时域特征') > -1){
      features.value = ['均值', '方差', '标准差', '峰度', '偏度', '四阶累积量', '六阶累积量', '最大值', '最小值', '中位数', '峰峰值', '整流平均值', '均方根', '方根幅值',
        '波形因子', '峰值因子', '脉冲因子', '裕度因子']
    }else if (nodeInfo.label_display.indexOf('频域特征') > -1){
      features.value = ['重心频率', '均方频率', '均方根频率', '频率方差', '频率标准差', '谱峭度的均值', '谱峭度的峰度']
    }
  }
  console.log(nodeInfo)
  //算法模块不允许重复
  if (nodeList.value.length === 0) {
    nodeList.value.push(nodeInfo)
  } else {
    let isDuplicate = false;
    for (let i = 0; i < nodeList.value.length; i++) {
      let nod = nodeList.value[i];
      if (nod.id == node.id) {
        // window.alert('不允许出现重复模块');
        ElMessage({
          message: '不允许出现同一类别的算法',
          type: 'warning'
        })
        isDuplicate = true;
        break;
      }
    }
    // 向节点列表中添加新拖拽入可视化建模区中的模块
    if (!isDuplicate) {
      nodeList.value.push(nodeInfo);
    }
  }

  // 将节点初始化为可以连线的状态
  nextTick(() => {
    plumbIns.draggable(nodeId, { containment: "efContainer" })
 
    if (node.id < 4) {
      plumbIns.makeSource(nodeId, deff.jsplumbSourceOptions)
    }

    plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)

  })
}

// 删除节点
const deleteNode = (nodeId) => {
  if (!modelSetup.value) {
    nodeList.value = nodeList.value.filter(node => node.id !== nodeId);
    plumbIns.deleteEveryConnection()
    plumbIns.deleteEveryEndpoint()
    linkedList.head = null
    linkedList.tail = null
    canCheckModel.value = true
    canStartProcess.value = true
    canShutdown.value = true
    canSaveModel.value = true
  }
}


// 处理可视化建模区中拖拽节点的操作
const handleMouseup = (ev, data) => { // 在图表中拖拽节点时，设置他的新的位置
  // nodeList.value.forEach(node => {
  //   if (node.id === data.id) {
  //     data.nodeContainerStyle.left = ev.clientX
  //     data.nodeContainerStyle.top = ev.clientY
  //     // node.value.nodeContainerStyle.left = ev.clientX + 'px'
  //     // node.value.nodeContainerStyle.top = ev.clientY + 'px'
  //   }
  // })
  if (!done.value) {
    length = nodeList.value.length
    for (let i = 0; i < length; i++) {
      let node = nodeList.value[i]
      if (node.id === data.id) {
        // setTimeout(()=>{
        //   data.nodeContainerStyle.left = ev.clientX - 290 
        //   data.nodeContainerStyle.top = ev.clientY - 80 
        // }, 2)
        nodeList.value[i].nodeContainerStyle.left = ev.clientX - 300 + 'px'
        nodeList.value[i].nodeContainerStyle.top = ev.clientY - 100 + 'px'
      }
    }
  }

}

const modelsetting = () => {
  selects.value = !selects.value
}

const dialogFormVisible = ref(false)  // 控制对话框弹出，输入要保存的模型的名称

// 提交的模型相关信息
const modelInfoForm = ref({
  name: '',
  region: '',
})

// 检查模型参数设置
const checkModelParams = () => {
  for (let i = 0; i < nodeList.value.length; i++) {
    let dict = nodeList.value[i]

    if (!dict.use_algorithm) 
    {
      console.log('')
      return false
    } 
    else 
    {
      // 检查特征选择的规则参数
      if (dict.id == '1.3'){
        let threshold = false
        // 检查选择特征的规则参数是否正确设置
        let rule = dict.parameters[dict.use_algorithm].rule
        if (dict.parameters[dict.use_algorithm]['threshold'+rule]){
          threshold = true
        }
        if (!threshold){
          return false
        }
      }
      else if (dict.id == '1.2'){
        if (!features.value.length) {
          return false
        }
      } 
    }
  }

  return true
}

//保存模型并取消拖拽动作                 
const saveModelSetting = (saveModel, schedule) => {

  done.value = true

  // dialogFormVisible.value = true
  // selects.value = !selects.value
  jsonClear()
  for (let i = 0; i < nodeList.value.length; i++) {
    let dict = nodeList.value[i]

    if (!dict.use_algorithm) {
      ElMessage({
        message: '请设置每个算法的必选属性',
        type: 'error'
      })
      console.log('dict.use_algorithm is empty! return')
      return
    }

    contentJson.algorithms[dict.label] = dict.use_algorithm
    if (!contentJson.modules.includes(dict.label)) {
      contentJson.modules.push(dict.label);
    }

    // 选择特征提取需要展示的参数
    if (dict.id == '1.2') {
      let params = dict.parameters[dict.use_algorithm]
      if (!features.value.length) {
        ElMessage({
          message: '请设置每个算法的必选属性',
          type: 'error'
        })
        return
      }
      features.value.forEach(element => {
        if (params[element] == false) {
          params[element] = true
        }
      });
      contentJson.parameters[dict.use_algorithm] = params
      continue
    }
    contentJson.parameters[dict.use_algorithm] = dict.parameters[dict.use_algorithm]
    // console.log(dict.use_algorithm + '\'s params are: ' + dict.parameters[dict.use_algorithm])

  }
  if (!modelCheckRight && saveModel) {
    ElMessage({
      message: '请先建立模型并通过模型检查！',
      type: 'warning'
    })
    return
  }
  let current = linkedList.head;
  contentJson.schedule.length = 0
  console.log('nodeList: ', nodeList.value)
  // 如果只有一个节点，无需建立流程
  if (nodeList.value.length == 1) {
    contentJson.schedule.push(nodeList.value[0].label)
  } else {
    if (!saveModel) {
      console.log('schedule: ', schedule)
      contentJson.schedule = schedule
      console.log('content_json: ', contentJson)
    } else {
      if (!current) {
        ElNotification({
          title: 'WARNING',
          message: '未建立流程，请先建立流程',
          type: 'warning',
        })
        return
      }
    }
    while (current) {
      contentJson.schedule.push(current.value);
      current = current.next;
    }
  }
  dialogModle.value = saveModel   
}

// 完成模型名称等信息的填写后，确定保存模型
const saveModelConfirm = () => {
  let data = new FormData()
  data.append('model_name', modelInfoForm.value.name)
  let nodelistInfo = nodeList.value
  let modelInfo = { "nodeList": nodelistInfo, "connection": contentJson.schedule }
  data.append('model_info', JSON.stringify(modelInfo))
  // data.append('username', window.localStorage.getItem('username'))
  // axios.post('http://127.0.0.1:8000/save_model/', data,
  //   {
  //     headers: { "Content-Type": 'multipart/form-data' }
  //   }
  // ).then((response) => {
  //   if (response.data.message == 'save_model_successful') {
  //     ElMessage({
  //       message: '保存模型成功',
  //       type: 'success'
  //     })
  //     fetch_models()
  //     models_drawer.value = false       // 关闭历史模型抽屉
  //     dialogFormVisible.value = false    // 关闭提示窗口
  //     dialogmodle.value = false
  //     canStartProcess.value = false     // 保存模型成功可以运行
  //     modelSetup.value = true                 // 模型保存完成
  //     updateStatus('当前模型已保存')
  //   } else {
  //     ElMessage({
  //       message: '保存模型失败',
  //       type: 'error'
  //     })
  //   }
  // })
  api.post('/user/save_model/', data,
    {
      headers: { "Content-Type": 'multipart/form-data' }
    }
  ).then((response) => {
    if (response.data.message == 'save model success') {
      ElMessage({
        message: '保存模型成功',
        type: 'success'
      })
      fetchModels()
      modelsDrawer.value = false       // 关闭历史模型抽屉
      dialogFormVisible.value = false    // 关闭提示窗口
      dialogModle.value = false
      canStartProcess.value = false     // 保存模型成功可以运行
      modelSetup.value = true                 // 模型保存完成
      modelLoaded.value = modelInfoForm.value.name  // 保存模型后，显示当前模型名称
      updateStatus('当前模型已保存')
    } else if(response.data.code == 400) {
      ElMessage({
        message: '已有同名模型，保存模型失败',
        type: 'error'
      })
    }
  }).catch(error=>{
    ElMessage({
      message: '保存模型请求失败',
      type: 'error'
    })
    console.log('save model error: ', error)
  })
}

const show1 = ref(false)

// 结果可视化区域显示
const canShowResults = ref(false)

// 健康评估结果展示
const healthEvaluation = ref('')
const displayHealthEvaluation = ref(false)
const activeName1 = ref('first')
const healthEvaluationFigure1 = ref('data:image/png;base64,')
const healthEvaluationFigure2 = ref('data:image/png;base64,')
const healthEvaluationFigure3 = ref('data:image/png;base64,')

const healthEvaluationDisplay = (results_object) => {

  displayHealthEvaluation.value = true
  let figure1 = results_object.层级有效指标_Base64
  let figure2 = results_object.二级指标权重柱状图_Base64
  let figure3 = results_object.评估结果柱状图_Base64

  healthEvaluation.value = results_object.评估建议
  healthEvaluationFigure1.value = 'data:image/png;base64,' + figure1
  healthEvaluationFigure2.value = 'data:image/png;base64,' + figure2
  healthEvaluationFigure3.value = 'data:image/png;base64,' + figure3

  // const imgElement1 = document.getElementById('health_evaluation_figure_1'); 
  // const imgElement2 = document.getElementById('health_evaluation_figure_2');  
  // const imgElement3 = document.getElementById('health_evaluation_figure_3');   
  // imgElement1.src = `data:image/png;base64,${figure1}`; 
  // imgElement2.src = `data:image/png;base64,${figure2}`; 
  // imgElement3.src = `data:image/png;base64,${figure3}`; 

}


// 特征提取结果展示
const displayFeatureExtraction = ref(false)
const transformedData = ref([])
const columns = ref([])

const featureExtractionDisplay = (resultsObject) => {

  displayFeatureExtraction.value = true
  // 获取后端传回的提取的特征
  let featuresWithName = Object.assign({}, resultsObject.features_with_name)
  let featuresName = featuresWithName.features_name.slice()
  let featuresGroupBySensor = Object.assign(featuresWithName.features_extracted_group_by_sensor)
  let datas = []        // 表格中每一行的数据
  featuresName.unshift('传感器')  // 表格的列名
  for (const sensor in featuresGroupBySensor) {
    let featuresOfSensor = featuresGroupBySensor[sensor].slice()
    featuresOfSensor.unshift(sensor)
    datas.push(featuresOfSensor)
  }
  columns.value.length = 0
  featuresName.forEach(element => {
    columns.value.push({ prop: element, label: element, width: 180 })
  });

  // 转换数据为对象数组  
  transformedData.value = datas.map((row, index) => {
    const obj = {};
    columns.value.forEach((column, colIndex) => {
      obj[column.prop] = row[colIndex];
    });
    return obj;
  });
 
}


// 特征选择结果可视化
const displayFeatureSelection = ref(false)
const featureSelectionFigure = ref('')
const featuresSelected = ref('')
const selectFeatureRule = ref('')

const featuresSelectionDisplay = (resultsObject) => {
  displayFeatureSelection.value = true

  let figure1 = resultsObject.figure_Base64
  featuresSelected.value = resultsObject.selected_features.join('、')
  selectFeatureRule.value = resultsObject.rule
  featureSelectionFigure.value = 'data:image/png;base64,' + figure1

}


// 故障诊断结果展示
const displayFaultDiagnosis = ref(false)
const faultDiagnosis = ref('')
const faultDiagnosisFigure = ref('')

const faultDiagnosisDisplay = (resultsObject) => {
  displayFaultDiagnosis.value = true

  let figure1 = resultsObject.figure_Base64
  let diagnosisResult = resultsObject.diagnosis_result
  if (diagnosisResult == 0) {
    faultDiagnosis.value = '无故障'
  } else {
    faultDiagnosis.value = '存在故障'
  }
  faultDiagnosisFigure.value = 'data:image/png;base64,' + figure1

}


// 故障预测结果展示
const displayFaultRegression = ref(false)
const haveFault = ref(0)
const faultRegression = ref('')
const timeToFault = ref('')
const faultRegressionFigure = ref('')

const faultRegressionDisplay = (resultsObject) => {
  displayFaultRegression.value = true

  let figure1 = resultsObject.figure_Base64
  faultRegressionFigure.value = 'data:image/png;base64,' + figure1
  // let fault_time = results_object.time_to_fault

  if (resultsObject.time_to_fault == 0) {
    haveFault.value = 1
    faultRegression.value = '已经出现故障'
  } else {
    haveFault.value = 0
    faultRegression.value = '还未出现故障'
    timeToFault.value = resultsObject.time_to_fault_str
  }

}

// 插值处理可视化
const activeName3 = ref('1')
const displayInterpolation = ref(false)
const interpolationFigures = ref([]) // 插值处理结果图像
const interpolationResultsOfSensors = ref([])   // 后端插值处理返回结果

const interpolationDisplay = (resultsObject: any) => {
  displayInterpolation.value = true

  let sensorId = 0
  interpolationFigures.value.length = 0
  interpolationResultsOfSensors.value.length = 0
  for(const [key, value] of Object.entries(resultsObject)){
    sensorId += 1
    interpolationFigures.value.push('data:image/png;base64,' + value)
    interpolationResultsOfSensors.value.push({label: key.split('_')[0], name: sensorId.toString()})
  }
  console.log('interpolationResultsOfSensors: ', interpolationResultsOfSensors)
  console.log('interpolationFigures: ', interpolationFigures)
  // displayDenoise.value = true
}

// 无量纲化可视化
const activeName4 = ref('1')
const displayNormalization = ref(false)
const normalizationFormdataRaw = ref([])
const normalizationFormdataScaled = ref([])  // 无量纲的结果表格
const normalizationColumns = ref([])
const normalizationResultFigures = ref([])   // 无量纲结果图像
const normalizationResultsSensors = ref([])

const transformDataToFormdata = (features_with_name: any, columns: any, formdata: any) => {

  let featuresName = features_with_name.features_name.slice()
  let featuresGroupBySensor = Object.assign(features_with_name.features_extracted_group_by_sensor)
  let datas = []        // 表格中每一行的数据
  featuresName.unshift('传感器')  // 表格的列名
  for (const sensor in featuresGroupBySensor) {
    let features_of_sensor = featuresGroupBySensor[sensor].slice()
    features_of_sensor.unshift(sensor)
    // console.log('features_of_sensor: ', features_of_sensor)
    datas.push(features_of_sensor)
    // features_of_sensor.splice(0, 1)
  }
  // console.log('features_name: ', features_name)
  // console.log('datas: ', datas)
  // let i = { prop: '', label: '', width: '' }
  columns.value.length = 0
  featuresName.forEach(element => {
    // console.log('element: ', element)
    columns.value.push({ prop: element, label: element, width: 180 })
  });

  // console.log('columns: ', columns)
  // 转换数据为对象数组  
  formdata.value = datas.map((row, index) => {
    const obj = {};
    columns.value.forEach((column, colIndex) => {
      obj[column.prop] = row[colIndex];
    });
    return obj;
  });
}

const normalizationResultType = ref('table')   // 无量纲化的结果类型，table表示表格，figure表示图像

const normalizationDisplay = (resultsObject: any) => {
  displayNormalization.value = true

  let rawData = Object.assign({}, resultsObject.raw_data)
  let scaledData = Object.assign({}, resultsObject.scaled_data)

  if (resultsObject.datatype == 'table') {
    normalizationResultType.value = 'table'
    transformDataToFormdata(rawData, normalizationColumns, normalizationFormdataRaw)
    transformDataToFormdata(scaledData, normalizationColumns, normalizationFormdataScaled)
  }
  else {
    normalizationResultType.value = 'figure'
    let sensorId = 0
    normalizationResultFigures.value.length = 0
    normalizationResultsSensors.value.length = 0
    for(const [key, value] of Object.entries(resultsObject)){
      if (key=='datatype') {
        continue
      }
      sensorId += 1
      normalizationResultFigures.value.push('data:image/png;base64,' + value)
      normalizationResultsSensors.value.push({label: key.split('_')[0], name: sensorId.toString()})
    }
  }
  
  // console.log('normalization_formdata_raw: ', normalizationFormdataRaw)
  // console.log('normalization_formdata_scaled: ', normalizationFormdataScaled)
  // console.log('normalization_columns: ', normalizationColumns)

}

// 小波降噪可视化
const activeName2 = ref('1')
const displayDenoise = ref(false)
const denoiseFigures = ref([])  // 存放小波降噪结果图片
const waveletResultsOfSensors = ref([])  // 存放不同传感器的小波降噪结果

const denoiseDisplay = (resultsObject) => {
  console.log('results_object: ', resultsObject)
  let sensorId = 0
  denoiseFigures.value.length = 0
  waveletResultsOfSensors.value.length = 0
  for(const [key, value] of Object.entries(resultsObject)){
    sensorId += 1
    denoiseFigures.value.push('data:image/png;base64,' + value)
    waveletResultsOfSensors.value.push({label: key.split('_')[0], name: sensorId.toString()})
  }
  console.log('results_of_sensors: ', waveletResultsOfSensors)
  console.log('denoiseFigures: ', denoiseFigures)
  displayDenoise.value = true
  // denoiseFigure.value = 'data:image/png;base64,' + results_object.sensor1_figure_Base64
}

// 清除可视化区域
const resultsViewClear = () => {
  showPlainIntroduction.value = false  // 清除算法介绍
  showStatusMessage.value = false      // 清除程序运行状态
  canShowResults.value = false         // 清除可视化区域元素
  show1.value = true
  loading.value = true
  isShow.value = false
  // 清除所有结果可视化
  displayHealthEvaluation.value = false
  displayFeatureExtraction.value = false
  displayFeatureSelection.value = false
  displayFaultDiagnosis.value = false
  displayFaultRegression.value = false
  displayInterpolation.value = false
  displayNormalization.value = false
  displayDenoise.value = false
}


// 点击可视化建模区中的算法模块显示对应的结果
const resultShow = (item) => {

  if (done.value) {

    if (missionComplete.value) {
      if (item.label != '层次分析模糊综合评估' && item.label != '特征提取' && item.label != '特征选择' && item.label != '故障诊断'
        && item.label != '趋势预测' && item.label != '特征提取' && item.label != '插值处理' && item.label != '无量纲化' && item.label != '小波变换'
      ) {
        showPlainIntroduction.value = false
        showStatusMessage.value = false
        show1.value = true
        loading.value = true
        canShowResults.value = false
        isShow.value = false
        setTimeout(function () {
          isShow.value = true
          show1.value = false
          loading.value = false
        }, 2500); 
        let moduleName = item.label
        let url = 'http://127.0.0.1:8000/homepage?display=' + moduleName
        axios.request({
          method: 'GET',
          url: url,
        });
        setTimeout(function () {
          // 为 iframe 的 src 属性添加一个查询参数，比如当前的时间戳，以强制刷新
          var iframe = document.getElementById('my_gradio_app');
          var currentSrc = iframe.src;
          var newSrc = currentSrc.split('?')[0]; // 移除旧的查询参数
          iframe.src = newSrc + '?updated=' + new Date().getTime();
        }, 2400);
      } else {
        resultsViewClear()
        canShowResults.value = true
        if (item.label == '层次分析模糊综合评估') {
          let results_to_show = responseResults.层次分析模糊综合评估
          healthEvaluationDisplay(results_to_show)
        } else if (item.label == '特征提取') {
          let results_to_show = responseResults.特征提取
          featureExtractionDisplay(results_to_show)
        } else if (item.label == '特征选择') {
          let results_to_show = responseResults.特征选择
          featuresSelectionDisplay(results_to_show)
        } else if (item.label == '故障诊断') {
          let results_to_show = responseResults.故障诊断
          faultDiagnosisDisplay(results_to_show)
        } else if (item.label == '趋势预测') {
          let results_to_show = responseResults.趋势预测
          faultRegressionDisplay(results_to_show)
        } else if (item.label == '插值处理') {
          let results_to_show = responseResults.插值处理
          interpolationDisplay(results_to_show)
        } else if (item.label == '无量纲化') {
          let results_to_show = responseResults.无量纲化
          normalizationDisplay(results_to_show)
        } else if (item.label == '小波变换') {
          let results_to_show = responseResults.小波变换
          denoiseDisplay(results_to_show)
        }
        else {
          ElMessage({
            message: '无效的算法模块',
            type: 'error'
          })
        }
      }
    }
  } else {
    // ElMessage({
    //   message: '当前无运行结果',
    //   type: 'error'
    // })
  }
}


// 打开抽屉，同时从后端获取历史模型
const fetchModels = () => {
  dataDrawer.value = false
  modelsDrawer.value = true
  // 向后端发送请求获取用户的历史模型
  api.get('/user/fetch_models/').then((response) => {
    let modelsInfo = response.data
    fetchedModelsInfo.value.length = 0
    for (let item of modelsInfo) {
      fetchedModelsInfo.value.push(item)
    }
  })
}

// 从后端获取到的历史模型的信息
const fetchedModelsInfo = ref([])


// 复用历史模型，不需要进行模型检查等操作
let modelHasBeenSaved = false
const modelLoaded = ref('无')  // 已加载的历史模型

// 点击历史模型表格中使用按钮触发复现历史模型
const useModel = (row) => {

  if (nodeList.value.length != 0) {
    nodeList.value.length = 0
  }
  handleClear()
  updateStatus('当前模型已保存')
  modelHasBeenSaved = true
  canStartProcess.value = false
  modelLoaded.value = row.model_name
  let objects = JSON.parse(row.model_info)
  let nodeList1 = objects.nodeList         // 模型节点信息   
  let connection = objects.connection      // 模型连线信息

  // 恢复节点
  for (let node of nodeList1) {

    nodeList.value.push(node)

    if (node.label == '特征提取') {
      features.value.length = 0
      let params = node.parameters[node.use_algorithm]
      for (let [key, value] of Object.entries(params)) {
        if (value) {
          features.value.push(key)
        }
      }
    }
  }
  // 用于将节点的id与节点的label对应起来
  let idToLabelList = { 'nodeId': [], 'nodeLabel': [] }

  // 初始化每个节点的可连接状态
  for (let node of nodeList.value) {

    let nodeId = node.id
    idToLabelList.nodeId.push(nodeId)
    idToLabelList.nodeLabel.push(node.label)

    nextTick(() => {
      // plumbIns.draggable(nodeId, { containment: "efContainer" })
      if (node.id === '2.2') {
        plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)
        return
      }
      plumbIns.makeSource(nodeId, deff.jsplumbSourceOptions)
      // plumbIns.addEndpoint(nodeId, deff.jsplumbTargetOptions)
      if (node.id === '1') {
        return
      }
      plumbIns.makeTarget(nodeId, deff.jsplumbTargetOptions)
    })
  }

  // 根据返回的模型的连接顺序，恢复模型中的连线
  let connectionList = []
  let connection2 = []   // 记录每个节点的id
  let node_num = connection.length

  for (let i = 0; i < node_num; i++) {
    let label = connection[i]
    for (let j = 0; j < node_num; j++) {
      if (idToLabelList.nodeLabel[j] === label) {
        connection2[i] = idToLabelList.nodeId[j]
        break
      }
    }
  }

  saveModelSetting(false, connection)
  contentJson.schedule = connection
  console.log('conten_json3: ', contentJson)
  modelSetup.value = true
  // 如果只有一个节点，则不恢复连线，否则按照模型信息中各模块的连接顺序恢复连线
  if (node_num == 1) {
    connectionList = []
  } else {
    for (let i = 0; i < node_num - 1; i++) {
      connectionList.push({ 'soruce_id': connection2[i], 'target_id': connection2[i + 1] })
    }
    nextTick(() => {
      for (let line of connectionList) {
        plumbIns.connect({
          source: document.getElementById(line.soruce_id),
          target: document.getElementById(line.target_id)
        })
      }
    })
  }
}


let index = 0
let row = 0
const deleteModelConfirmVisible = ref(false)
// 删除模型操作
const deleteModel = (indexIn, rowIn) => {
  index = indexIn
  row = rowIn
  deleteModelConfirmVisible.value = true
}
// 用户删除模型操作确认
const deleteModelConfirm = () => {

  // 发送删除请求到后端，row 是要删除的数据行
  api.get('/user/delete_model/?row_id=' + row.id).then((response) => {
    if (response.data.message == 'deleteSuccessful') {
      if (index !== -1) {
        // 删除前端表中数据
        fetchedModelsInfo.value.splice(index, 1)
        deleteModelConfirmVisible.value = false
      }
    }
  }).catch(error => {
    // 处理错误，例如显示一个错误消息  
    console.error(error);
  });
}

// 查看模型的具体信息，按如下方式构造信息卡片
const modelName = ref('')
const modelAlgorithms = ref([])
const modelParams = ref([])  // {'模块名': xx, '算法': xx, '参数': xx}

const showModelInfo = (row) => {
  let objects = JSON.parse(row.model_info)
  let nodesList = objects.nodeList         // 模型节点信息   
  let connection = objects.connection     // 模型连接顺序

  modelName.value = row.model_name
  modelAlgorithms.value = connection
  modelParams.value.length = 0
  nodesList.forEach(element => {
    let item = { '模块名': '', '算法': '' }
    item.模块名 = element.label
    item.算法 = labelsForAlgorithms[element.use_algorithm]
    modelParams.value.push(item)
  });
}

// 用于显示程序运行的状态信息
const showStatusMessage = ref(false)
const statusMessageToShow = ref('')

// 程序运行状态信息
const statusMessage = {
  'success': '## 程序已经运行完毕，请点击相应的算法模块查看对应结果！',
  'shutdown': '## 程序运行终止，点击清空模型重新建立模型',
  'error': '## 程序运行出错，请检查模型是否正确，或者检查加载的数据是否规范，点击清空模型重新建立模型',
}


// 控制是否可以修改模型，值为true时，可以修改模型，值为false时，不能修改模型
const toRectifyModel = ref(false)

// 完成建模
const finishModeling = () => {
  if (nodeList.value.length) {
    if (linkedList.length() == 0 && nodeList.value.length == 1) {
      ElMessage({
        message: '完成建模',
        type: 'success'
      })
      canCheckModel.value = false 
      modelSetup.value = true     // 不能删除建模区的模块
      done.value = true           // 不能拖动模块
      toRectifyModel.value = true // 可以点击修改模型进行修改模型
      updateStatus('模型建立完成但未通过检查')
      return
    }
    if (linkedList.length() != nodeList.value.length) {
      ElMessage({
        message: '请确保图中所有模块均已建立连接，且没有多余的模块',
        type: 'warning'
      })
      return
    }
  }

  ElMessage({
    message: '完成建模',
    type: 'success'
  })
  modelSetup.value = true     // 不能删除建模区模块
  done.value = true     // 不能拖动模块
  toRectifyModel.value = true // 可以修改模型
  canCheckModel.value = false
  updateStatus('模型建立完成但未通过检查')
}

// 修改模型
const rectifyModel = () => {
  canCheckModel.value = true
  canSaveModel.value = true
  canStartProcess.value = true
  canShutdown.value = true
  modelSetup.value = false     // 可以删除建模区模块
  done.value = false     // 可以拖动模块
  toRectifyModel.value = false
  ElMessage({
    showClose: true,
    message: '进行模型修改, 完成修改后请再次点击完成建模',
    type: 'info'
  })
  updateStatus('正在修改模型')
}

//检查模型
const checkModeling = () => {
  if (nodeList.value.length == 0 && !modelHasBeenSaved) {
    canCheckModel.value = true
  }
}

//保存模型
const saveModeling = () => {
  if (nodeList.value.length == 0 || modelHasBeenSaved) {
    canSaveModel.value = true
  }
}

//开始建模
const startModeling = () => {
  if (nodeList.value.length == 0) {
    canStartProcess.value = true
  }
}


// 建模状态更新
function updateStatus(status) {
  var indicator = document.getElementById('statusIndicator');
  indicator.textContent = status; // 更新文本  
  indicator.classList.remove('error', 'success', 'saved', 'rectify'); // 移除之前的状态类  
  switch (status) {
    case '未建立模型':
      // 默认样式，或者设置为特定类  
      break;
    case '模型建立完成但未通过检查':
      indicator.classList.add('error');
      break;
    case '模型建立并已通过模型检查':
      indicator.classList.add('success');
      break;
    case '当前模型已保存':
      indicator.classList.add('saved');
    case '正在修改模型':
      indicator.classList.add('rectify')
      break;
  }
}


const fetchedDataFiles = ref<Object[]>([])

// 用户目前选择的数据文件
const usingDatafile = ref('无')

const deleteDatasetConfirmVisible = ref(false)
let rowDataset: any = null
let indexDataset: any = null
// 用户删除历史数据
const deleteDataset = (index_in: any, row_in: any) => {
  indexDataset = index_in
  rowDataset = row_in
  deleteDatasetConfirmVisible.value = true
}

const deleteDatasetConfirm = () => {

  api.get('/user/delete_datafile?filename=' + rowDataset.dataset_name)
    .then((response: any) => {
      if (response.data.code == 200){
        // 删除前端表中数据
        fetchedDataFiles.value.splice(indexDataset, 1)
        deleteDatasetConfirmVisible.value = false
        ElMessage({
          message: '文件删除成功',
          type: 'success'
        })
      }else if(response.data.code == 400){
        ElMessage({
          message: '删除失败: ' + response.data.message,
          type: 'error'
        })
      }
     
    })
    .catch(error => {
      console.log('delete_datafile_error: ', error)
      ElMessage({
        message: '删除失败',
        type: 'error'
      })
    })
}


const loadingData = ref(false)
// 用户选择历史数据
const useDataset = (row_in: any) => {
  loadingData.value = true
  setTimeout(() => {
    loadingData.value = false
    usingDatafile.value = row_in.dataset_name
    ElMessage({
      message: '数据加载成功',
      type: 'success'
    })
}, 1000)
  
}

const handleSwitchDrawer = (fetchData: any[]) => {

  modelsDrawer.value = false;

  fetchedDataFiles.value = []
  // fetchData.forEach(element => {
  //   fetchedDataFiles.value.push(element)
  // });
  for (let item of fetchData){
    fetchedDataFiles.value.push(item)
  }
  nextTick(() => {
    console.log('nextTick')
    console.log('fetchedDataFiles: ', fetchedDataFiles.value)
  })

  dataDrawer.value = true
  
  // Object.assign(fetchedDataFiles, fetchData)
  console.log("fetchData: ", fetchData);
  console.log("fetchedDataFiles: ", fetchedDataFiles.value);
};
</script>

<style>
body {
  margin: 0;
}


.item {
  width: 150px;
  height: 50px;
  position: relative;
}

.deleteButton {
  position: absolute;
  top: 0px;
  right: 0px;
}

#source {
  border: 2px solid red;
}

#target {
  border: 2px solid blue;
}

.main {
  display: flex;
}

ul {
  list-style: none;
  padding-left: 0;
  width: 120px;
  background: #eee;
  text-align: center;
}

ul>li {
  height: 40px;
  line-height: 40px;
}

.main-right {
  border: 1px solid #ccc;
  flex: 1;
  margin-left: 15px;
  position: relative;
  background: #f4f4f4;
}

.node-info {
  position: relative;
  top: 5px
}

.node-info-label {
  /* font-style: italic; */
  /* 垂直和水平居中的样式 */
  display: flex;
  align-items: center;
  justify-content: center;
  overflow-wrap: break-word;
  /* font-style: italic; */
  padding: 4px;
  position: relative;
  width: 112px;
  height: 80px;
  /* line-height: 36px; */
  font-size: 16px;
  text-align: center;
  border: 1px solid #e5e7eb;
  background: #fff;
  border-radius: 4px;
}

.node-info-label:hover {

  cursor: pointer;
  background: #f4eded;
}

.node-info-label:hover+.node-drag {
  /* background: red; */
  display: inline-block;
}

.node-drag {
  display: none;
  width: 10px;
  height: 10px;
  border-radius: 10px;
  background-color: red;
  border: 1px solid #ccc;
  position: absolute;
  right: -10px;
  top: 40px;
}

.node-drag:hover {
  display: inline-block;
}

.fullscreen_container {
  height: 100vh;
  /* display: flex; */
  /* flex-direction: column; */
}

.result_visualization_view {
  width: 1200px;
  height: 600px;
  position: absolute;
}

.demo-tabs .custom-tabs-label span {
  vertical-align: middle;
  font-size: 16px;
  margin-left: 9px;
}

.status-indicator {
  position: fixed;
  top: 70px;
  right: 10px;
  padding: 5px 10px;
  border-radius: 5px;
  background-color: #f0ad4e;
  /* 初始颜色，如黄色 */
  color: white;
  z-index: 1000;
  /* 确保它显示在其他元素之上 */
}

/* 可以为不同的状态添加额外的类 */
.status-indicator.error {
  background-color: #d9534f;
  /* 红色表示错误或未通过检查 */
}

.status-indicator.success {
  background-color: #5cb85c;
  /* 绿色表示成功 */
}

.status-indicator.saved {
  background-color: #337ab7;
  /* 蓝色表示已保存 */
}

.status-indicator.rectify {
  background-color: #48a4a3;
  /* 表示正在修改模型 */
}

html,
body {
  height: 100vh;
  margin: 0;
  /* 移除默认的边距 */
  padding: 0;
  /* 移除默认的内边距 */
}

.el-main {
  background-color: #cee1f6;
  /*color: #333;*/
  text-align: center;
  position: relative;
  background-position: center;
  background-size: contain;
  /* height: 50vh; */
  background-repeat: no-repeat;
}

.has-background {
  background-color: #cee1f6;
  /*color: #333;*/
  text-align: center;
  position: relative;
  background-image: url('../assets/modeling.png');
  background-position: center;
  background-size: contain;
  /* height: 50vh; */
  background-repeat: no-repeat;
}

.clickable:hover {
  cursor: pointer;
  color: #007BFF;
}

.aside-title {
  font-size: 20px; 
  font-weight: 700;
  background-color: #1F5EBA; 
  width: 250px; 
  color: #f9fbfa;
}

.first-menu-item{
  width: 150px; 
  margin-top: 10px; 
  background-color: #4C74DA; 
  color: white;
}

.second-menu-item{
  width: 150px; 
  margin-top: 7px; 
  background-color: #7E9CE6;
}

.third-menu-item{
  background-color: #B9BFCE ; 
  margin-top: 7px; 
  width: 145px; 
  height: 30px; 
  margin-bottom: 10px; 
  padding: 0px; 
  border-radius: 5px; 
  align-content: center; 
  margin-left: 40px;
}
</style>
