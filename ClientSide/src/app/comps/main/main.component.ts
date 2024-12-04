import { Component, ElementRef, SecurityContext, ViewChild } from '@angular/core';
import { TextService } from 'src/app/servers/text.service';
import { DomSanitizer } from '@angular/platform-browser';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-main',
  templateUrl: './main.component.html',
  styleUrls: ['./main.component.css']
})
export class MainComponent {
  @ViewChild('inputText') inputText: ElementRef | undefined;

  constructor(private sc:TextService,private http: HttpClient){}
  t:string=''
  loader=false
  text:string=''
  fileOutput: string | null = null;
  parts:string=''
  cleaned_str:string=''
  
  selectedFile: File | undefined;
  fileContent: string = '';
  
  IfText:boolean= true;
   
  clearParagraph() {
    if (this.text === '') {
      this.t = '';
    }
  }
  errorMessage: string | null = null;

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file: File = input.files[0];
      const reader = new FileReader();
      const fileType = file.type;
      const fileName = file.name;

      reader.onload = (e) => {
        this.fileContent = reader.result as string;
        console.log(this.fileContent); 
        this.text=this.fileContent; // הדפסת תוכן הקובץ
      };
      if (fileType !== 'text/plain' || !fileName.endsWith('.txt')) {
        this.errorMessage = 'Please upload a TXT file only.';
        // איפוס שדה הקובץ
        input.value = '';
        this.text='';
      } else {
        this.errorMessage = null;
        reader.readAsText(file);

      }
     

    }
    
  }
  
    
  

  



  gett()
  {  
      if(this.text!=''){
        this.loader=true
        this.sc.translate(this.text).subscribe(
          secc=>{this.t= JSON.stringify(secc);
            
            ;
            this.loader=false
            const jsonObject = JSON.parse(this.t);

// חילוץ הטקסט בצורה יותר ממוקדת
           const translationText = jsonObject.translation[0].translation_text;

           console.log(translationText)
           const extractedText = translationText.split("'")[3].trim();
          //  console.log(extractedText.split('"')[3].trim());
          const extractedText2 = extractedText.split('"')[0].trim();
           console.log(extractedText)

           this.t=extractedText2;
          }
          ,err=>{console.log(err);
            this.loader=false

          }
          
        )
      }
  }






}

