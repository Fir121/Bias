from nicegui import ui
import pandas as pd
from bias_modules.llm_calls import ModelHandler, Constants
import asyncio
from bias_modules.stat import class_balance_checker, chi_square_test, text_length_classifier_deviation, get_accuracy, demographic_parity
import joblib
import os
import markdown

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras

# GLOBALS
df = text_col = label_col = ddesc = None
display_output = False

model = model_type = model_desc = results_df = text_col_p = label_col_p = pred_col = None
display_output_2 = False

message_history = []
message_history_2 = []

def pandas_to_table(dataframe):
    return ui.  table(
        columns=[{'name': col, 'label': col, 'field': col} for col in dataframe.columns],
        rows=dataframe.to_dict('records'),
    )

def get_modelinfo_from_path(path, model_type):
    if model_type == "SciKit":
        mdl = joblib.load(path)
        data = str(mdl) + "\n\n"
        for step in mdl.named_steps:
            data += f"Step: {step}\n"
            data += f"{mdl.named_steps[step].get_params()}\n"
        return data
    elif model_type == "TensorFlow" or model_type == "Keras":
        mdl = keras.models.load_model(path)
        return mdl.summary()
    elif model_type == "Custom":
        return model_desc

def header_bar():
    with ui.header().classes('items-baseline justify-between'):
        ui.label('DetBias').classes('text-3xl')
        ui.label('A bias detection and mitigation tool by developers, for developers').classes('text-md')


async def stage1():
    global df, text_col, label_col, ddesc

    with ui.stepper().props('vertical').classes('w-full') as stepper:
        with ui.step('Dataset Upload'):
            ui.label('Upload your training dataset (.csv)')

            def handle_file_upload(f):
                global df
                ui.notify(f'Uploaded {f.name}')
                nb1.enable()
                inp.disable()
                with open('temp.csv', 'wb') as file:
                    file.write(f.content.read())
                df = pd.read_csv('temp.csv')
                colselect.refresh()
            
            def handle_file_reject():
                ui.notify('Rejected!')
                nb1.disable()
                inp.enable()

            def clear():
                inp.reset()
                nb1.disable()
                inp.enable()
            
            inp = ui.upload(
                on_upload=handle_file_upload,
                on_rejected=handle_file_reject,
                max_file_size=1_000_000_000,
                auto_upload=True,
                multiple=False,
                max_files=1,
            ).classes('max-w-full').props("accept=.csv")

            ui.button('Clear Files', on_click=clear).classes('bg-red-500 text-white').props('flat')

            with ui.stepper_navigation():
                nb1 = ui.button('Next', on_click=stepper.next)
                nb1.disable()
            
        with ui.step('Dataset Columns'):
            @ui.refreshable
            async def colselect():
                global text_col, label_col

                if df is not None:
                    col_list = df.columns.tolist()

                    loop = asyncio.get_running_loop()
                    temp = ui.spinner(size='lg')

                    text_col_llm = label_col_llm = None
                    if not text_col or not label_col:
                        llm = ModelHandler()
                        prompt1 = Constants.get_col_name(col_list, 'column containing text value for classification')
                        prompt2 = Constants.get_col_name(col_list, 'column containing the label value for classification')

                        text_col_llm = await loop.run_in_executor(None, llm.generate_response, prompt1)
                        label_col_llm = await loop.run_in_executor(None, llm.generate_response, prompt2)

                    def check_vals():
                        global text_col, label_col
                        if text_col.value and label_col.value:
                            nb2.enable()
                        else:
                            nb2.disable()

                    temp.delete()

                    ui.label('Select the column containing the text')
                    text_col = ui.select(options=col_list, on_change=check_vals, value=text_col_llm if text_col_llm in col_list else None)

                    ui.label('Select the label column')
                    label_col = ui.select(options=col_list, on_change=check_vals, value=label_col_llm if label_col_llm in col_list else None)

                    def move_next():
                        stepper.next()
                        desc_entry.refresh()

                    with ui.stepper_navigation():
                        nb2 = ui.button('Next', on_click=move_next)
                        nb2.disable()
                        check_vals()
                        ui.button('Back', on_click=stepper.previous).props('flat')
                else:
                    ui.label('Upload a dataset first!')
            
            await colselect()

        with ui.step('Dataset Description'):
            @ui.refreshable
            async def desc_entry():
                global text_col, label_col, ddesc

                loop = asyncio.get_running_loop()
                temp = ui.spinner(size='lg')

                if not ddesc and text_col and label_col and text_col.value and label_col.value:
                    llm = ModelHandler()
                    prompt = Constants.get_classifier_determination_system_prompt(label_col.value, text_col.value, df[label_col.value].unique().tolist())
                    ddesc = await loop.run_in_executor(None, llm.generate_response, prompt)

                def set_val(val):
                    global ddesc
                    ddesc = val.value

                temp.delete()

                ui.label('What is the purpose of this dataset?')
                ta = ui.textarea(value=ddesc, on_change=set_val).classes('w-1/2')

                def final_submit():
                    global display_output
                    display_output = True
                    ui.notify("Good to go! Please wait a bit...", type='positive')
                    stage1_switcher.refresh()

                with ui.stepper_navigation():
                    ui.button('Process Bias', on_click=final_submit).bind_enabled_from(ta, 'value')
                    ui.button('Back', on_click=stepper.previous).props('flat')
            
            await desc_entry()

async def output_stage_1():
    def cancel():
        global display_output
        display_output = False
        stage1_switcher.refresh()
    
    ui.button('Reset', on_click=cancel).classes('bg-red-500 text-white').props('flat')

    load = ui.spinner(size='lg').classes('mx-auto')
    ll = ui.label('Processing, please wait, this might take a while...').classes('text-center w-full')

    loop = asyncio.get_running_loop()

    class_dist, balanced = await loop.run_in_executor(None, class_balance_checker, df, label_col.value)
    chi_df = await loop.run_in_executor(None, chi_square_test, ddesc, df, text_col.value, label_col.value)
    length_df, lbalanced = await loop.run_in_executor(None, text_length_classifier_deviation, df, text_col.value, label_col.value)

    load.delete()
    ll.delete()

    if balanced:
        ui.label('✅ The dataset is balanced')
    else:
        ui.label('❌ The dataset is imbalanced')
        ui.echart({
            'type': 'bar',
            'data': {
                'labels': class_dist.index.tolist(),
                'datasets': [{
                    'label': 'Count of rows in dataset',
                    'data': class_dist.values.tolist(),
                }],
            },
        })

    if chi_df["response"].str.contains("INVALID").any():
        invalid_words = ', '.join(chi_df[chi_df['response'].str.contains('INVALID', na=False)]['word'].tolist())
        ui.label(f'❌ The dataset contains textual bias [Chi Square Test] | Potential biased words: {invalid_words}')
        pandas_to_table(chi_df)
    else:
        ui.label('✅ The dataset does not contain textual bias [Chi Square Test]')

    if lbalanced:
        ui.label('✅ The dataset is balanced in terms of text length')
    else:
        ui.label('❌ The dataset is imbalanced in terms of text length')
        pandas_to_table(length_df)

    prompt = Constants.pre_analysis_prompt(ddesc, text_col.value, label_col.value, balanced, class_dist, chi_df, lbalanced, length_df)
    await chat_dialog(prompt, 1)

@ui.refreshable
async def stage1_switcher():
    if not display_output:
        await stage1()
    else:
        await output_stage_1()

async def stage2():
    global model, model_type, model_desc, results_df, text_col_p, label_col_p, pred_col

    with ui.stepper().props('vertical').classes('w-full') as stepper:
        with ui.step('Predictions Upload'):
            ui.label('Upload your predictions (.csv)')

            def handle_file_upload2(f):
                global results_df
                ui.notify(f'Uploaded {f.name}')
                nb12.enable()
                inp2.disable()
                with open('temp2.csv', 'wb') as file:
                    file.write(f.content.read())
                results_df = pd.read_csv('temp2.csv')
            
            def handle_file_reject2():
                ui.notify('Rejected!')
                nb12.disable()
                inp2.enable()

            def clear2():
                inp2.reset()
                nb12.disable()
                inp2.enable()
            
            inp2 = ui.upload(
                on_upload=handle_file_upload2,
                on_rejected=handle_file_reject2,
                max_file_size=1_000_000_000,
                auto_upload=True,
                multiple=False,
                max_files=1,
            ).classes('max-w-full').props("accept=.csv")

            ui.button('Clear Files', on_click=clear2).classes('bg-red-500 text-white').props('flat')

            def nxt():
                colselect.refresh()
                stepper.next()

            with ui.stepper_navigation():
                nb12 = ui.button('Next', on_click=nxt)
                nb12.disable()
            
        with ui.step('Predictions Columns'):
            @ui.refreshable
            async def colselect():
                global text_col_p, label_col_p, pred_col, results_df

                if results_df is not None:
                    col_list2 = results_df.columns.tolist()

                    loop = asyncio.get_running_loop()
                    temp = ui.spinner(size='lg')

                    text_col_llm = label_col_llm = pred_col_llm = None
                    if not text_col_p or not label_col_p or not pred_col:
                        llm = ModelHandler()
                        prompt1 = Constants.get_col_name(col_list2, 'column containing text value for classification')
                        prompt2 = Constants.get_col_name(col_list2, 'column containing the label value for classification')
                        prompt3 = Constants.get_col_name(col_list2, 'column containing the predictions')

                        text_col_llm = await loop.run_in_executor(None, llm.generate_response, prompt1)
                        label_col_llm = await loop.run_in_executor(None, llm.generate_response, prompt2)
                        pred_col_llm = await loop.run_in_executor(None, llm.generate_response, prompt3)

                    def check_vals():
                        global text_col_p, label_col_p, pred_col
                        if text_col_p.value and label_col_p.value and pred_col.value:
                            nb22.enable()
                        else:
                            nb22.disable()

                    temp.delete()

                    ui.label('Select the column containing the text')
                    text_col_p = ui.select(options=col_list2, on_change=check_vals, value=text_col_llm if text_col_llm in col_list2 else None)

                    ui.label('Select the label column')
                    label_col_p = ui.select(options=col_list2, on_change=check_vals, value=label_col_llm if label_col_llm in col_list2 else None)

                    ui.label('Select the predictions column')
                    pred_col = ui.select(options=col_list2, on_change=check_vals, value=pred_col_llm if pred_col_llm in col_list2 else None)

                    def move_next():
                        stepper.next()

                    with ui.stepper_navigation():
                        nb22 = ui.button('Next', on_click=move_next)
                        nb22.disable()
                        check_vals()
                        ui.button('Back', on_click=stepper.previous).props('flat')
                else:
                    ui.label('Upload a dataset first!')
            
            await colselect()

        with ui.step('Model Upload'):
            ui.label('Upload your model (.pkl, .h5, .pth)')

            def handle_file_upload(f):
                global model
                ui.notify(f'Uploaded {f.name}')
                nb1.enable()
                inp.disable()
                fn = 'model.'+f.name.split(".")[-1]
                with open(fn, 'wb') as file:
                    file.write(f.content.read())
                model = fn
            
            def handle_file_reject():
                ui.notify('Rejected!')
                nb1.disable()
                inp.enable()

            def clear():
                inp.reset()
                nb1.disable()
                inp.enable()
            
            inp = ui.upload(
                on_upload=handle_file_upload,
                on_rejected=handle_file_reject,
                max_file_size=1_000_000_000,
                auto_upload=True,
                multiple=False,
                max_files=1,
            ).classes('max-w-full').props("accept=.pkl, .h5, .pth")

            ui.button('Clear Files', on_click=clear).classes('bg-red-500 text-white').props('flat')

            def nxt2():
                colselect2.refresh()
                stepper.next()

            with ui.stepper_navigation():
                nb1 = ui.button('Next', on_click=nxt2)
                nb1.disable()

        with ui.step('Model Type'):
            @ui.refreshable
            async def colselect2():
                global model_type, model_desc

                if results_df is not None and text_col_p and label_col_p and pred_col and model:
                    types = ["SciKit", "TensorFlow", "Keras", "Custom"]

                    def check_vals():
                        global model_type
                        if model_type.value == "Custom":
                            edl.set_visibility(True)
                            model_desc.set_visibility(True)
                            if model_desc.value:
                                nb2.enable()
                            else:
                                nb2.disable()
                        elif model_type.value:
                            edl.set_visibility(False)
                            model_desc.set_visibility(False)
                            nb2.enable()
                        else:
                            edl.set_visibility(False)
                            model_desc.set_visibility(False)
                            nb2.disable()

                    ui.label('Select the type of the mode:')
                    model_type = ui.select(options=types, on_change=check_vals)

                    edl = ui.label('Describe the model - you may paste a technical description from model information')
                    model_desc = ui.textarea(on_change=check_vals).classes('w-1/2')

                    edl.set_visibility(False)
                    model_desc.set_visibility(False)

                    def move_next22():
                        global display_output_2
                        display_output_2 = True
                        ui.notify("Good to go! Please wait a bit...", type='positive')
                        stage2_switcher.refresh()

                    with ui.stepper_navigation():
                        nb2 = ui.button('Process Bias', on_click=move_next22)
                        nb2.disable()
                        ui.button('Back', on_click=stepper.previous).props('flat')
                else:
                    ui.label('Upload a dataset first!')
            
            await colselect2()

@ui.refreshable
async def chat_dialog(initial_prompt, usage):
    global message_history, message_history_2

    llm = ModelHandler()
    loop = asyncio.get_running_loop()
    if usage == 1:
        if not message_history:
            message_history = [
                {"role": "system", "content": Constants.detbias_sysprompt()},
                {"role": "user", "content": initial_prompt},
            ]
    else:
        if not message_history_2:
            message_history_2 = [
                {"role": "system", "content": Constants.detbias_sysprompt()},
                {"role": "user", "content": initial_prompt},
            ]
        
    c_his = message_history if usage == 1 else message_history_2

    with ui.card(align_items="stretch").classes('w-full'):
        async def send_message():
            message = ip.value
            bt.disable()
            c_his.append({"role": "user", "content": message})
            chat_dialog.refresh()

        with ui.column():
            ip = ui.input(placeholder='Type your message...').classes('w-full')
            bt = ui.button('Send', on_click=send_message)
            bt.disable()

        with ui.scroll_area().classes('w-full h-[40vh]'):
            with ui.row():
                load = ui.spinner(size='lg').classes('mx-auto')

                for message in c_his[2:][::-1]:
                    if message['role'] == 'user':
                        ui.chat_message(message['content'], name='user', avatar='https://robohash.org/super2', sent=True).classes('w-full')
                    else:
                        ui.label("DetBias' Response").classes('text-sm text-gray-500')
                        ui.markdown(message['content']).classes('w-full')
                
                if c_his[-1]['role'] == 'assistant':
                    load.delete()
                    bt.enable()
                else:
                    response = await loop.run_in_executor(None, llm.message_ai, c_his)
                    c_his.append({"role": "assistant", "content": response})
                    load.delete()
                    chat_dialog.refresh()
        

async def output_stage_2():
    def cancel():
        global display_output_2
        display_output_2 = False
        stage2_switcher.refresh()
    
    ui.button('Reset', on_click=cancel).classes('bg-red-500 text-white').props('flat')

    load = ui.spinner(size='lg').classes('mx-auto')
    ll = ui.label('Processing, please wait, this might take a while...').classes('text-center w-full')

    loop = asyncio.get_running_loop()

    model_desc = await loop.run_in_executor(None, get_modelinfo_from_path, model, model_type.value)
    accuracy = await loop.run_in_executor(None, get_accuracy, results_df, label_col_p.value, pred_col.value)
    demographic_parity_flag, df_std, demographic_parity_df = await loop.run_in_executor(None, demographic_parity, results_df, label_col_p.value, pred_col.value)

    load.delete()
    ll.delete()

    ui.label(f'Model Accuracy: {accuracy}%')

    if demographic_parity_flag:
        ui.label(f'❌ The model has a demographic parity issue with a standard deviation of {df_std}')
        pandas_to_table(demographic_parity_df)
    else:
        ui.label('✅ The model does not have a demographic parity issue')

    prompt = Constants.post_analysis_prompt(model_desc, model_type.value, accuracy, df_std, text_col_p.value, label_col_p.value, pred_col.value)
    await chat_dialog(prompt, 2)

@ui.refreshable
async def stage2_switcher():
    if not display_output_2:
        await stage2()
    else:
        await output_stage_2()


@ui.page('/', dark=True)
async def index():
    header_bar()

    ui.label('Select your current stage of work, for further analysis:').classes('text-sm')
    with ui.tabs() as tabs:
        ui.tab('Data Processing')
        ui.tab('Model Testing')

    with ui.tab_panels(tabs, value='Data Processing').classes('w-full'):
        with ui.tab_panel('Data Processing').classes('w-full'):
            await stage1_switcher()

        with ui.tab_panel('Model Testing').classes('w-full'):
            await stage2_switcher()

ui.run()